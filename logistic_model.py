import json
import pickle as pkl
import random
import re
import os
import unicodedata
from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                            confusion_matrix, f1_score, precision_score,
                            recall_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
# Import multiprocessing for parallel processing
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
# Import all functions from create_graphs.py
from create_logistic_graphs import (
    should_filter_title, format_authors, add_period_variants, standardize_journal_name,
    process_articles_to_graph, process_articles_to_author_graphs, normalize_name, get_paper_id
)

# Function to extract features for paper pairs
def extract_features(paper1_id, paper2_id, G, author_coauthor_G):
    paper1 = G.nodes[paper1_id]
    paper2 = G.nodes[paper2_id]
    features = {}
    
    # 1. Number of shared authors
    authors1 = set(paper1.get('authors', '').split('; '))
    authors2 = set(paper2.get('authors', '').split('; '))
    features['shared_authors'] = len(authors1.intersection(authors2))
    
    # 2. Number of shared citations
    citations1 = set(G.successors(paper1_id)) if paper1_id in G else set()
    citations2 = set(G.successors(paper2_id)) if paper2_id in G else set()
    features['shared_citations'] = len(citations1.intersection(citations2))
    
    # 3. Two-hop connection indicator
    two_hop = False
    for citation in citations1:
        if citation in G:
            citation_refs = set(G.successors(citation))
            if paper2_id in citation_refs:
                two_hop = True
                break
    features['two_hop_connection'] = 1 if two_hop else 0
    
    # 4. Paper metadata similarity (title similarity)
    title1 = paper1.get('title', '')
    title2 = paper2.get('title', '')
    abstract1 = paper1.get('abstract', '')
    
    # TF-IDF for text similarity using title1, abstract1, and title2
    if (title1 or abstract1) and title2:
        # Combine title and abstract for paper1
        paper1_text = (title1 + " " + abstract1).strip()
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([paper1_text, title2])
            features['title_abstract_similarity'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            features['title_abstract_similarity'] = 0
    else:
        features['title_abstract_similarity'] = 0
    
    # 5. Common coauthor weight (using edge weights from coauthorship graph)
    common_coauthor_weight = 0
    
    # Convert author strings to the format used in the coauthorship graph
    graph_authors1 = []
    graph_authors2 = []
    
    for author in authors1:
        parts = author.split()
        if len(parts) >= 2:
            first_initial = parts[0][0].lower() if parts[0] else ''
            last_name = parts[-1].lower()
            if first_initial:
                graph_authors1.append(f"{first_initial}. {last_name}")
    
    for author in authors2:
        parts = author.split()
        if len(parts) >= 2:
            first_initial = parts[0][0].lower() if parts[0] else ''
            last_name = parts[-1].lower()
            if first_initial:
                graph_authors2.append(f"{first_initial}. {last_name}")
    
    # Check for common coauthors in the coauthorship graph and use edge weights
    total_coauthor_weight = 0
    total_author_pairs = len(graph_authors1) * len(graph_authors2)
    
    for author1 in graph_authors1:
        if author1 in author_coauthor_G:
            for author2 in graph_authors2:
                if author2 in author_coauthor_G:
                    # Check if these authors have common coauthors
                    coauthors1 = set(author_coauthor_G.neighbors(author1))
                    coauthors2 = set(author_coauthor_G.neighbors(author2))
                    common_coauthors = coauthors1.intersection(coauthors2)
                    
                    # Sum the weights of coauthorship connections
                    for common_coauthor in common_coauthors:
                        # Get weights from the coauthorship graph
                        weight1 = author_coauthor_G[author1][common_coauthor].get('weight', 1)
                        weight2 = author_coauthor_G[author2][common_coauthor].get('weight', 1)
                        total_coauthor_weight += weight1 + weight2
    # Scale by the number of pairwise coauthors
    if total_author_pairs > 0:
        common_coauthor_weight = total_coauthor_weight / total_author_pairs
    else:
        common_coauthor_weight = 0
    
    features['common_coauthor_weight'] = common_coauthor_weight
    
    # 6. Other citations shared citations
    # (main article cites papers that cite the same source as the candidate paper)
    other_shared_citations_count = 0
    
    # For each paper cited by the main article
    for citation in citations1:
        if citation in G:
            # Get what this citation cites
            citation_refs = set(G.successors(citation))
            
            # Get what the candidate paper cites
            candidate_refs = set(G.successors(paper2_id)) if paper2_id in G else set()
            
            # Count the overlap
            other_shared_citations_count += len(citation_refs.intersection(candidate_refs))
    
    features['other_shared_citations'] = other_shared_citations_count
    
    return features

# Function to process a batch of paper pairs in parallel
def process_paper_pairs_batch(batch_data):
    """
    Process a batch of paper pairs to extract features in parallel.
    
    Args:
        batch_data: Tuple containing (batch_pairs, G, author_coauthor_G)
            batch_pairs: List of (main_article, target_paper) pairs
            G: Citation graph
            author_coauthor_G: Author co-authorship graph
            
    Returns:
        List of (features, label, pair, metadata) tuples
    """
    batch_pairs, G, author_coauthor_G = batch_data
    results = []
    
    for main_article, target_paper, is_cited in batch_pairs:
        try:
            # Extract features
            features = extract_features(main_article, target_paper, G, author_coauthor_G)
            feature_values = list(features.values())
            
            # Set label (1 for cited, 0 for not cited)
            label = 1 if is_cited else 0
            
            # Create metadata for this pair
            source_data = {
                'paper_id': main_article,
                'title': G.nodes[main_article].get('title', ''),
                'authors': G.nodes[main_article].get('authors', []),
                'journal': G.nodes[main_article].get('journal', ''),
                'year': G.nodes[main_article].get('year', '')
            }
            
            target_data = {
                'paper_id': target_paper,
                'title': G.nodes[target_paper].get('title', ''),
                'authors': G.nodes[target_paper].get('authors', []),
                'journal': G.nodes[target_paper].get('journal', ''),
                'year': G.nodes[target_paper].get('year', '')
            }
            
            pair_metadata = {
                'source': source_data,
                'target': target_data
            }
            
            results.append((feature_values, label, (main_article, target_paper), pair_metadata))
        except Exception as e:
            print(f"Error processing pair ({main_article}, {target_paper}): {str(e)}")
            continue
    
    return results

# Prepare data for classification - using all negative examples
def prepare_classification_data(G, author_coauthor_G, max_pairs_per_main=500):
    """
    Prepare classification data by extracting features for paper pairs.
    
    Args:
        G: Citation graph
        author_coauthor_G: Author co-authorship graph
        max_pairs_per_main: Maximum number of pairs to process per main article
        
    Returns:
        X_np, y_np, paper_pairs, metadata_list
    """
    # Get main articles
    main_articles = [node for node in G.nodes() if G.nodes[node].get('is_main_article', True)]
    all_papers = list(G.nodes())
    
    print(f"Processing {len(main_articles)} main articles with {len(all_papers)} potential target papers")
    
    # Create data structures
    X = []
    y = []
    paper_pairs = []
    metadata_list = []
    
    # Track counts
    positive_count = 0
    negative_count = 0
    
    # Determine number of workers for parallel processing
    num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    print(f"Using {num_workers} workers for parallel processing")
    
    # Process main articles in batches
    batch_size = 10  # Number of main articles to process in each batch
    total_batches = (len(main_articles) + batch_size - 1) // batch_size
    
    progress_bar = tqdm(total=total_batches, desc="Processing batches of main articles")
    
    # Process batches of main articles
    for batch_idx in range(0, len(main_articles), batch_size):
        batch_main_articles = main_articles[batch_idx:batch_idx + batch_size]
        all_batch_pairs = []
        
        # Prepare all pairs for this batch
        for main_article in batch_main_articles:
            try:
                # Get papers that the main article cites
                cited_papers = list(G.successors(main_article))
                num_citations = len(cited_papers)
                
                # Add all positive examples (cited papers)
                for target_paper in cited_papers:
                    all_batch_pairs.append((main_article, target_paper, True))
                
                # Get all non-cited papers
                non_cited_papers = [p for p in all_papers if p != main_article and p not in cited_papers]
                
                # Sample exactly 5 times the number of positive examples
                num_negative_samples = min(len(non_cited_papers), 5 * num_citations)
                if len(non_cited_papers) > num_negative_samples:
                    non_cited_papers = random.sample(non_cited_papers, num_negative_samples)
                
                # Add negative examples
                for target_paper in non_cited_papers:
                    all_batch_pairs.append((main_article, target_paper, False))
                    
            except Exception as e:
                print(f"Error preparing pairs for article {main_article}: {str(e)}")
                continue
        
        # Split pairs into sub-batches for parallel processing
        sub_batch_size = max(1, len(all_batch_pairs) // (num_workers * 4))  # Smaller chunks for better load balancing
        sub_batches = [all_batch_pairs[i:i + sub_batch_size] for i in range(0, len(all_batch_pairs), sub_batch_size)]
        
        # Process sub-batches in parallel
        batch_results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all sub-batches for processing
            future_to_batch = {
                executor.submit(process_paper_pairs_batch, (sub_batch, G, author_coauthor_G)): sub_batch 
                for sub_batch in sub_batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    results = future.result()
                    batch_results.extend(results)
                except Exception as e:
                    print(f"Error in parallel processing: {str(e)}")
        
        # Process results from this batch
        for feature_values, label, pair, pair_metadata in batch_results:
            X.append(feature_values)
            y.append(label)
            paper_pairs.append(pair)
            metadata_list.append(pair_metadata)
            
            # Update counts
            if label == 1:
                positive_count += 1
            else:
                negative_count += 1
        
        # Update progress
        progress_bar.update(1)
        
        # Save intermediate results every 10 batches
        if (batch_idx // batch_size) % 10 == 0 and batch_idx > 0:
            print(f"\nSaving intermediate results after {batch_idx + batch_size} main articles...")
            intermediate_data = {
                'X': X,
                'y': y,
                'paper_pairs': paper_pairs,
                'metadata': metadata_list
            }
            # Save to cache folder with proper path
            os.makedirs('cache_logistic', exist_ok=True)
            with open(f'cache_logistic/pairs_intermediate_{batch_idx + batch_size}.pkl', 'wb') as f:
                pkl.dump(intermediate_data, f)
    
    progress_bar.close()
    
    # Convert to numpy arrays
    X_np = np.array(X)
    y_np = np.array(y)
    
    print(f"Total positive examples: {positive_count}")
    print(f"Total negative examples: {negative_count}")
    print(f"Ratio of negative to positive examples: {negative_count / positive_count:.2f}")
    
    return X_np, y_np, paper_pairs, metadata_list

def prepare_train_test_split(X, y, paper_pairs, metadata, hard_negative_threshold=0.05, 
                            negative_positive_ratio=10, hard_negative_ratio=0.9, 
                            test_size=0.01, random_seed=111):
    """
    Prepares training and testing datasets with balanced positive and negative examples.
    
    Args:
        X: Feature matrix
        y: Target labels (1 for positive, 0 for negative)
        paper_pairs: List of paper pairs
        metadata: List of metadata for each pair
        hard_negative_threshold: Threshold for title_abstract_similarity to consider a negative example as "hard"
        negative_positive_ratio: Ratio of negative to positive examples to keep
        hard_negative_ratio: Ratio of hard negatives to total negatives
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, pairs_train, pairs_test, metadata_train, metadata_test
    """
    # Identify positive and negative examples
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]

    # Calculate how many negative examples to keep
    n_positives = len(positive_indices)
    n_negatives_total = min(len(negative_indices), negative_positive_ratio * n_positives)

    # Identify hard negatives (title_abstract_similarity > threshold)
    # Feature index 3 corresponds to title_abstract_similarity
    hard_negative_mask = (y[negative_indices] == 0) & (X[negative_indices, 3] > hard_negative_threshold)
    hard_negative_indices = negative_indices[hard_negative_mask]
    normal_negative_indices = negative_indices[~hard_negative_mask]

    print(f"Found {len(hard_negative_indices)} hard negatives and {len(normal_negative_indices)} normal negatives")

    # Calculate how many of each type of negative to keep
    n_hard_negatives_to_keep = int(n_negatives_total * hard_negative_ratio)
    n_normal_negatives_to_keep = n_negatives_total - n_hard_negatives_to_keep

    # Sample from hard negatives
    np.random.seed(random_seed)
    n_hard_to_sample = min(len(hard_negative_indices), n_hard_negatives_to_keep)
    sampled_hard_negative_indices = np.random.choice(hard_negative_indices, n_hard_to_sample, replace=False)

    # Sample from normal negatives
    n_normal_to_sample = min(len(normal_negative_indices), n_normal_negatives_to_keep)
    sampled_normal_negative_indices = np.random.choice(normal_negative_indices, n_normal_to_sample, replace=False)

    # Combine positive and sampled negative indices
    keep_indices = np.concatenate([positive_indices, sampled_hard_negative_indices, sampled_normal_negative_indices])

    # Filter the data
    X_filtered = X[keep_indices]
    y_filtered = y[keep_indices]
    paper_pairs_filtered = [paper_pairs[i] for i in keep_indices]
    metadata_filtered = [metadata[i] for i in keep_indices]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, pairs_train, pairs_test, metadata_train, metadata_test = train_test_split(
        X_filtered, y_filtered, paper_pairs_filtered, metadata_filtered, 
        test_size=test_size, random_state=random_seed
    )

    # Count hard negatives in the final dataset
    hard_negatives_count = np.sum((y_filtered == 0) & (X_filtered[:, 3] > hard_negative_threshold))
    normal_negatives_count = np.sum((y_filtered == 0) & (X_filtered[:, 3] <= hard_negative_threshold))

    # Print statistics
    print(f"Dataset size: {len(X_filtered)}")
    print(f"Positive examples: {sum(y_filtered)}")
    print(f"Hard negative examples: {hard_negatives_count}, Ratio to positives: {hard_negatives_count / sum(y_filtered):.2f}")
    print(f"Normal negative examples: {normal_negatives_count}, Ratio to positives: {normal_negatives_count / sum(y_filtered):.2f}")
    print(f"Total negative examples: {len(y_filtered) - sum(y_filtered)}, Ratio to positives: {(len(y_filtered) - sum(y_filtered)) / sum(y_filtered):.2f}")
    print(f"Training set: {len(X_train)} samples, Positive: {sum(y_train)}, Negative: {len(y_train) - sum(y_train)}")
    print(f"Test set: {len(X_test)} samples, Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)}")
    
    return X_train, X_test, y_train, y_test, pairs_train, pairs_test, metadata_train, metadata_test

def plot_feature_distributions(X, feature_names, log_scale=True, figsize=(12, 10)):
    """
    Plot the distribution of features to examine if scaling is needed.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix where each column is a feature
    feature_names : list
        List of feature names corresponding to columns in X
    log_scale : bool, default=True
        Whether to use logarithmic scale for y-axis
    figsize : tuple, default=(12, 10)
        Figure size (width, height) in inches
    """
    
    # Create a figure to visualize feature distributions
    plt.figure(figsize=figsize)
    
    # Plot histograms for each feature in X
    for i, feature_name in enumerate(feature_names):
        plt.subplot(3, 2, i+1)
        sns.histplot(X[:, i], color='blue')
        plt.title(f'Distribution of {feature_name}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        if log_scale:
            plt.yscale('log')  # Set y-axis to logarithmic scale
    
    plt.tight_layout()
    plt.show()

def run_logistic_regression(X_train, X_test, y_train, y_test, pairs_test, G, metadata_test):
    """
    Run logistic regression model on the prepared data.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    X_test : numpy.ndarray
        Test feature matrix
    y_train : numpy.ndarray
        Training labels
    y_test : numpy.ndarray
        Test labels
    pairs_test : list
        List of paper pairs in the test set
    G : networkx.Graph
        Graph of paper relationships
    metadata_test : list
        Metadata for test set
        
    Returns:
    --------
    dict
        Dictionary containing model, predictions, and evaluation metrics
    """
    
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define feature names for better interpretation
    feature_names = ['shared_authors', 'shared_citations', 'two_hop_connection', 
                     'title_abstract_similarity', 'common_coauthor_weight', 
                     'other_citations_shared_citations']
    
    # Add a constant to the input features
    X_train_sm = sm.add_constant(X_train_scaled)
    X_test_sm = sm.add_constant(X_test_scaled)
    
    # Fit the logistic regression model using statsmodels
    sm_model = sm.Logit(y_train, X_train_sm).fit()
    
    # Print the summary table with labeled features
    print("\nDetailed Regression Statistics:")
    # Create a dictionary mapping positions to feature names (including the constant)
    feature_labels = ['const'] + feature_names
    print(sm_model.summary2(xname=feature_labels))
    
    # Get probability scores
    sm_probs = sm_model.predict(X_test_sm)
    
    # Plot histogram of probability scores
    plt.figure(figsize=(10, 6))
    plt.hist(sm_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Histogram of Statsmodels Logistic Regression Probability Scores (Test Set)')
    plt.xlabel('Probability Score')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    sm_preds_by_threshold = {}
    
    for threshold in thresholds:
        sm_preds_by_threshold[threshold] = (sm_probs >= threshold).astype(int)
        
        print(f"\nStatsmodels Logistic Regression Performance (threshold = {threshold}):")
        print(f"Accuracy: {accuracy_score(y_test, sm_preds_by_threshold[threshold]):.4f}")
        print(f"Precision: {precision_score(y_test, sm_preds_by_threshold[threshold]):.4f}")
        print(f"Recall: {recall_score(y_test, sm_preds_by_threshold[threshold]):.4f}")
        print(f"F1 Score: {f1_score(y_test, sm_preds_by_threshold[threshold]):.4f}")
        
        # Calculate Type I and Type II errors
        sm_tn, sm_fp, sm_fn, sm_tp = confusion_matrix(y_test, sm_preds_by_threshold[threshold]).ravel()
        sm_type1_error = sm_fp / (sm_fp + sm_tn)  # False positive rate (Type I error)
        sm_type2_error = sm_fn / (sm_fn + sm_tp)  # False negative rate (Type II error)
        print(f"Type I Error (False Positive Rate): {sm_type1_error:.4f}")
        print(f"Type II Error (False Negative Rate): {sm_type2_error:.4f}")
    
    # Create ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, sm_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    # Create a curve with false positives and false negatives
    # Calculate false positives and false negatives for different thresholds
    fp_values = []
    fn_values = []
    thresholds_list = np.linspace(0, 1, 100)
    for threshold in thresholds_list:
        preds = (sm_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fp_values.append(fp)
        fn_values.append(fn)
    
    # Normalize values
    total_negatives = sum(y_test == 0)
    total_positives = sum(y_test == 1)
    fp_normalized = [fp/total_negatives for fp in fp_values]
    fn_normalized = [fn/total_positives for fn in fn_values]
    
    # Plot FP vs FN curve with color-coded thresholds
    plt.figure(figsize=(10, 8))
    points = plt.scatter(fn_normalized, fp_normalized, c=thresholds_list, cmap='viridis', 
                        s=50, alpha=0.8, edgecolors='none')
    plt.colorbar(points, label='Threshold')
    
    # Add reference line
    plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Negatives (Type II Error)')
    plt.ylabel('False Positives (Type I Error)')
    plt.title('False Positives vs False Negatives Curve')
    plt.grid(alpha=0.3)
    plt.show()
    
    # Compare predictions on a random subset of 50 papers
    random_indices = random.sample(range(len(X_test)), 50)
    
    print("\nComparison of predictions on 50 random paper pairs with different thresholds:")
    print("Paper 1 Title | Paper 2 Title | Actual | SM(0.1) | SM(0.2) | SM(0.3) | SM(0.4) | SM(0.5) | SM Prob")
    print("-" * 140)
    
    for idx in random_indices:
        paper1_id, paper2_id = pairs_test[idx]
        paper1_title = G.nodes[paper1_id].get('title', '')[:30] + "..."
        paper2_title = G.nodes[paper2_id].get('title', '')[:30] + "..."
        
        actual = y_test[idx]
        
        threshold_preds = []
        for threshold in thresholds:
            threshold_preds.append(str(sm_preds_by_threshold[threshold][idx]))
        
        print(f"{paper1_title} | {paper2_title} | {actual} | {' | '.join(threshold_preds)} | {sm_probs[idx]:.4f}")
    
    return {
        'model': sm_model,
        'probabilities': sm_probs,
        'predictions_by_threshold': sm_preds_by_threshold,
        'scaler': scaler,
        'feature_names': feature_names,
        'roc_auc': roc_auc
    }

def find_citation_candidates(paper_title, G, author_coauthor_G, model, scaler=None, threshold=0.15, test_mode=False, test_fraction=0.25):
    """
    Find citation candidates for a specific paper.
    
    Parameters:
    -----------
    paper_title : str
        Title of the paper to find citation candidates for
    G : networkx.Graph
        Graph of paper relationships
    author_coauthor_G : networkx.Graph
        Graph of author co-authorship relationships
    model : statsmodels.discrete.discrete_model.BinaryResultsWrapper or sklearn model
        Trained model for citation prediction
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler used to standardize features
    threshold : float, default=0.15
        Probability threshold for recommending citations
    test_mode : bool, default=False
        Whether to run in test mode by temporarily removing some citations
    test_fraction : float, default=0.25
        Fraction of citations to remove in test mode
        
    Returns:
    --------
    tuple
        (results, paper_id, removed_citations)
    """
    
    # Find the paper ID by title
    paper_id = None
    for node in G.nodes():
        if G.nodes[node].get('title', '') == paper_title:
            paper_id = node
            break
    
    if paper_id is None:
        print(f"Paper with title '{paper_title}' not found in the graph.")
        return None, None, None
    
    print(f"Found paper ID: {paper_id}")
    print(f"Paper details: {G.nodes[paper_id]}")
    
    # Create a copy of the graph to modify for testing
    test_G = G.copy()
    
    # If in test mode, temporarily remove a fraction of the actual citations
    removed_citations = []
    if test_mode:
        actual_citations = list(G.successors(paper_id))
        if actual_citations:
            # Determine how many citations to remove
            num_to_remove = max(1, int(len(actual_citations) * test_fraction))
            # Randomly select citations to remove
            citations_to_remove = random.sample(actual_citations, num_to_remove)
            
            # Remove these edges from the test graph
            for citation in citations_to_remove:
                test_G.remove_edge(paper_id, citation)
                removed_citations.append(citation)
            
            print(f"Test mode: Temporarily removed {len(removed_citations)} citations out of {len(actual_citations)}")
    
    # Get all nodes except the target paper (any node can be a potential citation)
    potential_citations = [node for node in test_G.nodes() if node != paper_id]
    
    # Create feature vectors for all potential citation pairs
    candidate_features = []
    candidate_pairs = []
    
    print(f"Evaluating {len(potential_citations)} potential citation candidates...")
    
    for candidate_id in tqdm(potential_citations):
        # Extract features for this pair using the test graph
        features = extract_features(paper_id, candidate_id, test_G, author_coauthor_G)
        feature_values = list(features.values())
        candidate_features.append(feature_values)
        candidate_pairs.append((paper_id, candidate_id))
    
    # Convert to numpy array
    X_candidates = np.array(candidate_features)
    
    # Apply scaling if provided
    if scaler is not None:
        X_candidates = scaler.transform(X_candidates)
    
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        # For models like XGBoost that have predict_proba
        candidate_probs = model.predict_proba(X_candidates)[:, 1]
    else:
        # For statsmodels Logit models
        # Check if we need to add a constant term for statsmodels
        if X_candidates.shape[1] != len(model.params) - 1:  # -1 for the constant term
            # Add constant term if needed (statsmodels requires this)
            X_candidates = sm.add_constant(X_candidates)
        candidate_probs = model.predict(X_candidates)
    
    # Filter candidates based on threshold
    recommended_indices = np.where(candidate_probs >= threshold)[0]
    
    # Sort by probability (highest first)
    sorted_indices = recommended_indices[np.argsort(-candidate_probs[recommended_indices])]
    
    # Prepare results
    results = []
    for idx in sorted_indices:
        candidate_id = candidate_pairs[idx][1]
        candidate_title = G.nodes[candidate_id].get('title', '')
        candidate_authors = G.nodes[candidate_id].get('authors', '')
        candidate_journal = G.nodes[candidate_id].get('journal', '')
        candidate_year = G.nodes[candidate_id].get('year', '')
        probability = candidate_probs[idx]
        
        # Check if this is already cited in the original graph
        is_cited = candidate_id in G.successors(paper_id)
        
        # Check if this is one of the removed citations
        is_removed_citation = candidate_id in removed_citations
        
        results.append({
            'paper_id': candidate_id,
            'title': candidate_title,
            'authors': candidate_authors,
            'journal': candidate_journal,
            'year': candidate_year,
            'probability': probability,
            'already_cited': is_cited,
            'is_removed_citation': is_removed_citation
        })
    
    return results, paper_id, removed_citations

def evaluate_citation_recommendations(candidates, paper_id, G, removed_citations=None):
    """
    Evaluate the quality of citation recommendations.
    
    Parameters:
    -----------
    candidates : list
        List of candidate papers with their details
    paper_id : str
        ID of the paper for which recommendations were made
    G : networkx.Graph
        Graph of paper relationships
    removed_citations : list, optional
        List of citations that were removed for testing
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    if not candidates:
        print("No citation candidates found or paper not in database.")
        return None
    
    print(f"\nFound {len(candidates)} potential citation candidates:")
    print("-" * 100)
    
    # Sort by probability (highest first)
    for i, candidate in enumerate(candidates):
        citation_status = "✓ Already cited" if candidate['already_cited'] else "Not cited yet"
        removed_marker = "⚠ REMOVED FOR TESTING" if candidate['is_removed_citation'] else ""
        print(f"{i+1}. [{citation_status}] ({candidate['probability']:.4f}) {candidate['title']} {removed_marker}")
        print(f"   Authors: {candidate['authors']}")
        print(f"   Published in: {candidate['journal']} ({candidate['year']})")
        print("-" * 100)
    
    # Summarize citation coverage
    actual_citations = list(G.successors(paper_id))
    suggested_citations = [c['paper_id'] for c in candidates]
    
    # Count how many actual citations were suggested
    suggested_actual_citations = [cid for cid in actual_citations if cid in suggested_citations]
    
    metrics = {
        'total_actual_citations': len(actual_citations),
        'total_suggestions': len(suggested_citations),
        'suggested_actual_citations': len(suggested_actual_citations)
    }
    
    # If we're in test mode with removed citations
    if removed_citations:
        # Count how many removed citations were successfully recovered
        recovered_citations = [cid for cid in removed_citations if cid in suggested_citations]
        
        print("\nCitation Coverage Summary:")
        print(f"Total actual citations: {len(actual_citations)}")
        print(f"Citations temporarily removed for testing: {len(removed_citations)}")
        print(f"Removed citations that were successfully recovered: {len(recovered_citations)} ({len(recovered_citations)/len(removed_citations)*100:.1f}% recovery rate)")
        
        metrics['removed_citations'] = len(removed_citations)
        metrics['recovered_citations'] = len(recovered_citations)
        metrics['recovery_rate'] = len(recovered_citations)/len(removed_citations) if len(removed_citations) > 0 else 0
        
        # Regular metrics excluding the removed citations
        remaining_citations = [cid for cid in actual_citations if cid not in removed_citations]
        suggested_remaining = [cid for cid in remaining_citations if cid in suggested_citations]
        
        print(f"\nRegular citation metrics (excluding removed citations):")
        print(f"Suggested citations that are actual: {len(suggested_remaining)} ({len(suggested_remaining)/len(remaining_citations)*100:.1f}% recall)")
        
        metrics['remaining_citations'] = len(remaining_citations)
        metrics['suggested_remaining'] = len(suggested_remaining)
        metrics['recall_remaining'] = len(suggested_remaining)/len(remaining_citations) if len(remaining_citations) > 0 else 0
    else:
        # Regular evaluation without test mode
        recall = len(suggested_actual_citations) / len(actual_citations) if len(actual_citations) > 0 else 0
        print(f"\nCitation Coverage Summary:")
        print(f"Total actual citations: {len(actual_citations)}")
        print(f"Suggested citations that are actual: {len(suggested_actual_citations)} ({recall*100:.1f}% recall)")
        
        metrics['recall'] = recall
    
    if len(suggested_citations) > 0:
        # Calculate precision based on all suggestions
        if removed_citations:
            precision = (len(suggested_actual_citations)) / len(suggested_citations)
        else:
            precision = len(suggested_actual_citations) / len(suggested_citations)
        print(f"Precision: {precision*100:.1f}%")
        metrics['precision'] = precision
    
    # F1 score
    if 'precision' in metrics and (metrics['recall'] > 0 if 'recall' in metrics else metrics['recovery_rate'] > 0):
        if 'recall' in metrics:
            f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            # Use recovery rate as a proxy for recall in test mode
            f1 = 2 * (metrics['precision'] * metrics['recovery_rate']) / (metrics['precision'] + metrics['recovery_rate'])
        print(f"F1 Score: {f1*100:.1f}%")
        metrics['f1'] = f1
    
    return metrics

# Create cache directory if it doesn't exist
os.makedirs('cache_logistic', exist_ok=True)

# Step 0: Load articles
articles_cache_path = 'cache_logistic/articles.pkl'
if os.path.exists(articles_cache_path):
    print("Loading articles from cache...")
    with open(articles_cache_path, 'rb') as f:
        articles = pkl.load(f)
else:
    print("Loading articles from file...")
    with open('articles.jsonl', 'r') as f:
        articles = [json.loads(line) for line in f]
    # Cache articles
    with open(articles_cache_path, 'wb') as f:
        pkl.dump(articles, f)

# Step 1: Create citation graph from articles
citation_graph_cache_path = 'cache_logistic/citation_graph.pkl'
if os.path.exists(citation_graph_cache_path):
    print("Loading citation graph from cache...")
    with open(citation_graph_cache_path, 'rb') as f:
        G, main_article_count = pkl.load(f)
else:
    print("Creating citation graph...")
    G, main_article_count = process_articles_to_graph(articles)
    # Cache citation graph
    with open(citation_graph_cache_path, 'wb') as f:
        pkl.dump((G, main_article_count), f)

print(f"Main articles in graph: {main_article_count}")
print(f"Total nodes in graph: {len(G.nodes())}, edges: {len(G.edges())}")

# Add debugging code to check main article classification
print("\nDebugging main article classification:")
main_articles_in_graph = [node for node in G.nodes() if G.nodes[node].get('is_main_article', False)]
print(f"Number of nodes with is_main_article=True: {len(main_articles_in_graph)}")

# Check a sample of articles to see if they're properly classified
print("\nSample of 5 main articles:")
for i, node in enumerate(main_articles_in_graph[:5]):
    print(f"{i+1}. ID: {node}")
    print(f"   Title: {G.nodes[node].get('title', 'No title')}")
    print(f"   Journal: {G.nodes[node].get('journal', 'No journal')}")
    print(f"   Year: {G.nodes[node].get('year', 'No year')}")
    print(f"   Authors: {G.nodes[node].get('authors', 'No authors')}")
    print(f"   is_main_article: {G.nodes[node].get('is_main_article', False)}")
    print(f"   Number of references: {len(list(G.successors(node)))}")

# Check a sample of non-main articles
non_main_articles = [node for node in G.nodes() if not G.nodes[node].get('is_main_article', False)]
print("\nSample of 5 non-main articles (references):")
for i, node in enumerate(non_main_articles[:5]):
    print(f"{i+1}. ID: {node}")
    print(f"   Title: {G.nodes[node].get('title', 'No title')}")
    print(f"   Journal: {G.nodes[node].get('journal', 'No journal')}")
    print(f"   Year: {G.nodes[node].get('year', 'No year')}")
    print(f"   Authors: {G.nodes[node].get('authors', 'No authors')}")
    print(f"   is_main_article: {G.nodes[node].get('is_main_article', False)}")
    print(f"   Number of citations: {len(list(G.predecessors(node)))}")

# Check if there's a mismatch between main_article_count and actual main articles in graph
print(f"\nMain article count from function: {main_article_count}")
print(f"Main articles found in graph: {len(main_articles_in_graph)}")
if main_article_count != len(main_articles_in_graph):
    print("Mismatch between main_article_count and actual main articles in graph!")
    
    # Check the original articles - all articles in the input file are main articles
    print(f"Total articles in original data: {len(articles)}")
    
    # Check how many articles have valid title, date, and authors (requirements for inclusion)
    valid_articles = [a for a in articles if a.get('title') and a.get('published_date') and a.get('authors')]
    print(f"Valid articles in original data (with title, date, authors): {len(valid_articles)}")
    
    # Check if any main articles were filtered out during graph creation
    if len(valid_articles) > main_article_count:
        print(f"Some articles ({len(valid_articles) - main_article_count}) were filtered out during graph creation.")
        print("This could be due to issues with ID generation or duplicate IDs.")

# Add after the existing debugging code
print("\nChecking for potential ID collisions...")
# Count how many unique titles are in the original articles
unique_titles = set(a.get('title', '') for a in articles if a.get('title'))
print(f"Unique titles in original data: {len(unique_titles)}")

# Count how many unique IDs would be generated
article_ids = []
for article in articles:
    if article.get('title') and article.get('published_date') and article.get('authors'):
        article_id = get_paper_id(
            article.get('title'),
            article.get('authors', []),
            article['published_date'].split('-')[0]
        )
        if article_id:
            article_ids.append(article_id)

unique_ids = set(article_ids)
print(f"Unique IDs generated from original data: {len(unique_ids)}")
print(f"Difference between article count and unique IDs: {len(article_ids) - len(unique_ids)}")

if len(article_ids) > len(unique_ids):
    print("ID collisions detected - some articles are generating the same ID")
    # Find duplicate IDs
    from collections import Counter
    id_counts = Counter(article_ids)
    duplicate_ids = [id for id, count in id_counts.items() if count > 1]
    print(f"Number of duplicate IDs: {len(duplicate_ids)}")
    if len(duplicate_ids) > 0:
        print("Sample of duplicate IDs:")
        for dup_id in duplicate_ids[:3]:
            print(f"  ID: {dup_id}")
            # Find articles with this ID
            for article in articles:
                if (article.get('title') and article.get('published_date') and article.get('authors')):
                    article_id = get_paper_id(
                        article.get('title'),
                        article.get('authors', []),
                        article['published_date'].split('-')[0]
                    )
                    if article_id == dup_id:
                        print(f"    Title: {article.get('title')}")
                        print(f"    Authors: {article.get('authors')}")
                        print(f"    Year: {article.get('published_date')}")
                        print("    ---")

# After the ID collision check and before Step 2
print("\nAnalyzing reference articles in the graph...")
reference_articles = [node for node in G.nodes() if not G.nodes[node].get('is_main_article', False)]
print(f"Number of reference-only articles: {len(reference_articles)}")
print(f"Total nodes in graph: {len(G.nodes())}")
print(f"Percentage of reference-only articles: {len(reference_articles)/len(G.nodes())*100:.2f}%")

# Check how many references each main article has on average
reference_counts = [len(list(G.successors(node))) for node in main_articles_in_graph]
if reference_counts:
    avg_references = sum(reference_counts) / len(reference_counts)
    max_references = max(reference_counts)
    min_references = min(reference_counts)
    print(f"Average references per main article: {avg_references:.2f}")
    print(f"Maximum references for a main article: {max_references}")
    print(f"Minimum references for a main article: {min_references}")

# Check how many main articles cite each reference on average
citation_counts = [len(list(G.predecessors(node))) for node in reference_articles]
if citation_counts:
    avg_citations = sum(citation_counts) / len(citation_counts)
    max_citations = max(citation_counts)
    min_citations = min(citation_counts)
    print(f"Average citations per reference article: {avg_citations:.2f}")
    print(f"Maximum citations for a reference article: {max_citations}")
    print(f"Minimum citations for a reference article: {min_citations}")

# Check for references that are highly cited
highly_cited_threshold = 10
highly_cited = [node for node in reference_articles if len(list(G.predecessors(node))) >= highly_cited_threshold]
print(f"Number of highly cited references (≥{highly_cited_threshold} citations): {len(highly_cited)}")

if highly_cited:
    print("\nSample of highly cited references:")
    for i, node in enumerate(sorted(highly_cited, key=lambda x: len(list(G.predecessors(x))), reverse=True)[:5]):
        citation_count = len(list(G.predecessors(node)))
        print(f"{i+1}. ID: {node} (cited by {citation_count} main articles)")
        print(f"   Title: {G.nodes[node].get('title', 'No title')}")
        print(f"   Journal: {G.nodes[node].get('journal', 'No journal')}")
        print(f"   Year: {G.nodes[node].get('year', 'No year')}")
        print(f"   Authors: {G.nodes[node].get('authors', 'No authors')}")

# Step 2: Process all articles directly
author_graphs_cache_path = 'cache_logistic/author_graphs_all_articles.pkl'
if os.path.exists(author_graphs_cache_path):
    print("Loading author graphs from cache...")
    with open(author_graphs_cache_path, 'rb') as f:
        author_coauthor_G, author_citation_G, main_author_count = pkl.load(f)
else:
    print("Creating author graphs for all articles...")
    # Process all articles directly
    author_coauthor_G, author_citation_G, main_author_count = process_articles_to_author_graphs(articles)
    # Cache author graphs
    with open(author_graphs_cache_path, 'wb') as f:
        pkl.dump((author_coauthor_G, author_citation_G, main_author_count), f)

print(f"All articles - Main authors: {main_author_count}")
print(f"Author coauthorship edges: {len(author_coauthor_G.edges())}, citation edges: {len(author_citation_G.edges())}")

# Step 3: Prepare data for classification
classification_data_cache_path = 'cache_logistic/classification_data.pkl'
if os.path.exists(classification_data_cache_path):
    print("Loading classification data from cache...")
    with open(classification_data_cache_path, 'rb') as f:
        X, y, paper_pairs, metadata = pkl.load(f)
else:
    print("Preparing classification data...")
    X, y, paper_pairs, metadata = prepare_classification_data(G, author_coauthor_G)
    # Cache classification data
    with open(classification_data_cache_path, 'wb') as f:
        pkl.dump((X, y, paper_pairs, metadata), f)

# Step 4: Create train-test split with hard negatives
train_test_split_cache_path = 'cache_logistic/train_test_split.pkl'
if os.path.exists(train_test_split_cache_path):
    print("Loading train-test split from cache...")
    with open(train_test_split_cache_path, 'rb') as f:
        X_train, X_test, y_train, y_test, pairs_train, pairs_test, metadata_train, metadata_test = pkl.load(f)
else:
    print("Creating train-test split...")
    X_train, X_test, y_train, y_test, pairs_train, pairs_test, metadata_train, metadata_test = prepare_train_test_split(
        X, y, paper_pairs, metadata, 
        hard_negative_threshold=0.05,
        negative_positive_ratio=10,
        hard_negative_ratio=0.9,
        test_size=0.01
    )
    # Cache train-test split
    with open(train_test_split_cache_path, 'wb') as f:
        pkl.dump((X_train, X_test, y_train, y_test, pairs_train, pairs_test, metadata_train, metadata_test), f)

# Step 5: Plot feature distributions
feature_names = ['shared_authors', 'shared_citations', 'two_hop_connection', 
                 'title_abstract_similarity', 'common_coauthor_weight', 'other_shared_citations']
feature_distribution_cache_path = 'cache_logistic/feature_distribution_plot.pkl'
if os.path.exists(feature_distribution_cache_path):
    print("Loading feature distribution plot data from cache...")
    with open(feature_distribution_cache_path, 'rb') as f:
        plot_data = pkl.load(f)
else:
    print("Plotting feature distributions...")
    plot_feature_distributions(X_train, feature_names)
    # Cache feature distribution plot data
    with open(feature_distribution_cache_path, 'wb') as f:
        pkl.dump((X_train, feature_names), f)

# Step 6: Train and evaluate logistic regression model
model_results_cache_path = 'cache_logistic/model_results.pkl'
if os.path.exists(model_results_cache_path):
    print("Loading model results from cache...")
    with open(model_results_cache_path, 'rb') as f:
        results = pkl.load(f)
else:
    print("Training and evaluating logistic regression model...")
    results = run_logistic_regression(X_train, X_test, y_train, y_test, pairs_test, G, metadata_test)
    # Cache results
    with open(model_results_cache_path, 'wb') as f:
        pkl.dump(results, f)

