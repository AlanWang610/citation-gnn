import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os

# Load the saved paper citation graph
G = nx.read_gexf("JF_paper_metadata_citation_network.gexf")
print(f"Loaded citation network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Load the author coauthorship network
author_coauthor_G = nx.read_gexf('JF_author_coauthorship_network.gexf')

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

# Function to process a single main article
def process_main_article(args):
    main_article, all_papers, G, author_coauthor_G = args
    
    # Local data structures for this main article
    local_X = []
    local_y = []
    local_paper_pairs = []
    local_metadata_list = []
    local_positive_count = 0
    local_negative_count = 0
    
    # Get papers that the main article cites
    cited_papers = set(G.successors(main_article))
    
    # Process all possible target papers (excluding self)
    for target_paper in all_papers:
        if target_paper == main_article:
            continue
            
        # Extract features
        features = extract_features(main_article, target_paper, G, author_coauthor_G)
        feature_values = list(features.values())
        local_X.append(feature_values)
        
        # Set label (1 if cited, 0 if not cited)
        is_cited = 1 if target_paper in cited_papers else 0
        local_y.append(is_cited)
        
        # Track paper pair
        local_paper_pairs.append((main_article, target_paper))
        
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
        
        local_metadata_list.append(pair_metadata)
        
        # Update counts
        if is_cited:
            local_positive_count += 1
        else:
            local_negative_count += 1
    
    return {
        'X': local_X,
        'y': local_y,
        'paper_pairs': local_paper_pairs,
        'metadata': local_metadata_list,
        'positive_count': local_positive_count,
        'negative_count': local_negative_count
    }

# Prepare data for classification - using all possible data points
def prepare_classification_data(G, author_coauthor_G):
    
    # Get main articles
    main_articles = [node for node in G.nodes() if G.nodes[node].get('is_main_article', True)]
    all_papers = list(G.nodes())
    
    # Calculate total number of pairs to process
    total_pairs = len(main_articles) * (len(all_papers) - 1)  # Excluding self-citations
    print(f"Processing {len(main_articles)} main articles with {len(all_papers)} potential target papers")
    print(f"Total pairs to process: {total_pairs}")
    
    # Create data structures
    X = []
    y = []
    paper_pairs = []
    metadata_list = []
    
    # Track counts
    positive_count = 0
    negative_count = 0
    
    # Determine number of workers (use all available cores)
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} CPU cores for parallel processing")
    
    # Prepare arguments for parallel processing
    args_list = [(main_article, all_papers, G, author_coauthor_G) for main_article in main_articles]
    
    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress
        for result in tqdm(executor.map(process_main_article, args_list), 
                          total=len(main_articles), 
                          desc="Processing main articles",
                          position=0):
            results.append(result)
            
            # Accumulate intermediate results
            X.extend(result['X'])
            y.extend(result['y'])
            paper_pairs.extend(result['paper_pairs'])
            metadata_list.extend(result['metadata'])
            positive_count += result['positive_count']
            negative_count += result['negative_count']
            
            # Save intermediate results periodically
            if len(X) % 100000 == 0:
                print(f"\nSaving intermediate results at {len(X)} pairs...")
                intermediate_data = {
                    'X': X,
                    'y': y,
                    'paper_pairs': paper_pairs,
                    'metadata': metadata_list
                }
                with open(f'JF_pairs_intermediate_{len(X)}.pkl', 'wb') as f:
                    pickle.dump(intermediate_data, f)
    
    # Convert to numpy arrays
    X_np = np.array(X)
    y_np = np.array(y)
    
    print(f"Total positive examples: {positive_count}")
    print(f"Total negative examples: {negative_count}")
    print(f"Ratio of negative to positive examples: {negative_count / positive_count:.2f}")
    
    # Save the final results
    final_data = {
        'X': X_np,
        'y': y_np,
        'paper_pairs': paper_pairs,
        'metadata': metadata_list
    }
    
    with open('JF_pairs.pkl', 'wb') as f:
        pickle.dump(final_data, f)
    
    print(f"Data saved to JF_pairs.pkl")
    
    return X_np, y_np, paper_pairs, metadata_list

# Prepare the data (this will process with balanced sampling and save to file)
if __name__ == "__main__":
    X, y, paper_pairs, metadata = prepare_classification_data(G, author_coauthor_G)
