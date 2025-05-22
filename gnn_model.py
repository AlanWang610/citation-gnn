# Standard library imports
import json
import os
import pickle as pkl
import random
import re
import unicodedata
from collections import defaultdict
from itertools import product

# Third-party imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# For inductive model
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler

# Local imports
from create_gnn_graphs import (
    should_filter_title, format_authors, add_period_variants, 
    standardize_journal_name, normalize_name, get_paper_id, 
    format_author_name, get_journal_encoding, load_paper_embeddings, 
    process_articles_to_gnn_graph, save_gnn_graph, print_graph_statistics
)

# Global variables for the inductive model
global_tokenizer = None
global_model = None
global_year_scaler = None
global_embedding_dim = 768  # Default SciBERT embedding dimension

def load_scibert_model():
    """
    Load SciBERT model and tokenizer for embedding generation.
    
    Returns:
        Tuple of (tokenizer, model)
    """
    global global_tokenizer, global_model
    
    if global_tokenizer is None or global_model is None:
        print("Loading SciBERT model and tokenizer...")
        global_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        global_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    return global_tokenizer, global_model

def generate_scibert_embedding(text, tokenizer=None, model=None):
    """
    Generate SciBERT embedding for the given text.
    
    Args:
        text: Text to embed
        tokenizer: SciBERT tokenizer (optional)
        model: SciBERT model (optional)
        
    Returns:
        SciBERT embedding (numpy array)
    """
    # Load model if not provided
    if tokenizer is None or model is None:
        tokenizer, model = load_scibert_model()
    
    # Tokenize and embed
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]  # Get the [CLS] token embedding
    
    return embedding

def get_text_to_embed(title, abstract=None):
    """
    Combine title and abstract for embedding.
    
    Args:
        title: Paper title
        abstract: Paper abstract (optional)
        
    Returns:
        Combined text for embedding
    """
    if abstract and isinstance(abstract, str) and abstract.strip():
        return title + " " + abstract
    else:
        return title

def create_year_scaler(G):
    """
    Create a scaler for year normalization based on existing graph.
    
    Args:
        G: NetworkX graph with existing papers
        
    Returns:
        StandardScaler fitted on year data
    """
    global global_year_scaler
    
    # Extract years from graph
    years = []
    for node in G.nodes():
        year = G.nodes[node].get('year', 0)
        if isinstance(year, str) and year.isdigit():
            year = int(year)
        years.append(year)
    
    # Create and fit scaler
    years = np.array(years).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(years)
    
    # Store globally
    global_year_scaler = scaler
    
    return scaler

def generate_node_features_for_new_paper(paper_json, G, metadata=None, normalize_year=True):
    """
    Generate node features for a new paper in the same format as the existing graph.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph with existing papers
        metadata: Dictionary of paper metadata (optional)
        normalize_year: Whether to normalize the year feature
        
    Returns:
        Tuple of (node_id, node_features, paper_metadata)
    """
    # Extract paper information
    title = paper_json.get('title', '')
    year = paper_json.get('published_date', '').split('-')[0]
    journal = paper_json.get('journal', '')
    authors = paper_json.get('authors', [])
    abstract = paper_json.get('abstract', '')
    
    # Format authors
    formatted_authors = format_authors(authors)
    
    # Create paper ID
    paper_id = get_paper_id(title, authors, year)
    
    # Generate SciBERT embedding
    text_to_embed = get_text_to_embed(title, abstract)
    scibert_embedding = generate_scibert_embedding(text_to_embed)
    
    # Get journal encoding
    journal_encoding = get_journal_encoding(journal)
    
    # Get embedding dimension
    if G and len(G.nodes()) > 0:
        first_node = list(G.nodes())[0]
        global_embedding_dim = len(G.nodes[first_node]['scibert_embedding'])
    
    # Check if scibert_embedding needs to be resized
    if len(scibert_embedding) != global_embedding_dim:
        # If dimension doesn't match, we need to handle this
        # Options: 1) Truncate, 2) Pad with zeros, 3) Use dimensionality reduction
        if len(scibert_embedding) > global_embedding_dim:
            # Truncate
            scibert_embedding = scibert_embedding[:global_embedding_dim]
        else:
            # Pad with zeros
            padding = np.zeros(global_embedding_dim - len(scibert_embedding))
            scibert_embedding = np.concatenate([scibert_embedding, padding])
    
    # Convert year to int if possible
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    else:
        year = 0
    
    # Normalize year if requested
    if normalize_year and global_year_scaler is None and G:
        create_year_scaler(G)
    
    if normalize_year and global_year_scaler is not None:
        normalized_year = global_year_scaler.transform([[year]])[0][0]
    else:
        normalized_year = year
    
    # Create is_main_article flag (new papers are main articles)
    is_main_article = True
    
    # Create features array
    features = np.concatenate([
        np.array([normalized_year]).reshape(1),  # Year (normalized)
        np.array(journal_encoding),              # Journal encoding
        scibert_embedding,                       # SciBERT embedding
        np.array([1 if is_main_article else 0])  # Is main article
    ])
    
    # Create metadata dictionary
    paper_metadata = {
        'id': paper_id,
        'title': title,
        'year': year,
        'journal': journal,
        'authors': formatted_authors,
        'is_main_article': is_main_article
    }
    
    return paper_id, features, paper_metadata

def add_paper_to_graph(paper_json, G, metadata, paper_id=None, features=None, paper_metadata=None):
    """
    Add a new paper to the existing graph without retraining.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph with existing papers
        metadata: Dictionary of paper metadata
        paper_id: ID of the paper (optional, generated if not provided)
        features: Pre-computed features (optional)
        paper_metadata: Pre-computed metadata (optional)
        
    Returns:
        Updated graph and metadata
    """
    # Generate features if not provided
    if paper_id is None or features is None or paper_metadata is None:
        paper_id, features, paper_metadata = generate_node_features_for_new_paper(paper_json, G, metadata)
    
    # Check if paper already exists in graph
    if paper_id in G:
        print(f"Warning: Paper '{paper_id}' already exists in graph")
        return G, metadata
    
    # Add node to graph
    G.add_node(
        paper_id,
        year=paper_metadata['year'],
        journal_encoding=paper_metadata.get('journal_encoding', [0] * 8),  # Default encoding
        scibert_embedding=features[1+8:-1],  # Extract embedding from features
        is_main_article=paper_metadata['is_main_article']
    )
    
    # Add to metadata
    metadata[paper_id] = paper_metadata
    
    # Extract references from paper_json
    references = paper_json.get('references', [])
    
    # Add citation edges
    for ref in references:
        if ref.get('reference_type') != 'article':
            continue
            
        ref_title = ref.get('title')
        if not ref_title or should_filter_title(ref_title):
            continue
            
        ref_id = get_paper_id(
            ref_title,
            ref.get('authors', []),
            str(ref.get('year', ''))
        )
        
        if ref_id and ref_id in G:
            G.add_edge(paper_id, ref_id)
    
    return G, metadata

def inductive_recommend_citations(paper_json, G, metadata, model, node_features, 
                                node_mapping, reverse_mapping, author_coauthor_G=None, 
                                top_k=10, exclude_observed=True, device='cpu'):
    """
    Recommend citations for a new paper in an inductive setting.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph with existing papers
        metadata: Dictionary of paper metadata
        model: Trained EnhancedCitationMLP model
        node_features: Node features tensor for existing papers
        node_mapping: Mapping from node IDs to indices
        reverse_mapping: Mapping from indices to node IDs
        author_coauthor_G: Author co-authorship graph (optional)
        top_k: Number of top recommendations to return
        exclude_observed: Whether to exclude observed citations from recommendations
        device: Device to run the model on
        
    Returns:
        List of dictionaries containing recommended citations
    """
    # Extract observed citations
    observed_citations = []
    for ref in paper_json.get('references', []):
        if ref.get('reference_type') != 'article':
            continue
            
        ref_title = ref.get('title')
        if not ref_title or should_filter_title(ref_title):
            continue
            
        ref_id = get_paper_id(
            ref_title,
            ref.get('authors', []),
            str(ref.get('year', ''))
        )
        
        if ref_id and ref_id in G:
            observed_citations.append(ref_id)
    
    # Generate features for the new paper
    paper_id, paper_features, paper_metadata = generate_node_features_for_new_paper(paper_json, G, metadata)
    
    # Move model to device
    model = model.to(device)
    
    # Get candidate papers (all papers except the observed citations)
    candidates = [node for node in G.nodes()]
    
    # Exclude observed citations if requested
    if exclude_observed:
        candidates = [c for c in candidates if c not in observed_citations]
    
    # Convert paper features to tensor
    paper_features_tensor = torch.tensor(paper_features, dtype=torch.float).to(device)
    
    # Prepare batches for prediction
    batch_size = 1000
    all_probs = []
    
    # Process candidates in batches
    for i in range(0, len(candidates), batch_size):
        batch_candidates = candidates[i:i+batch_size]
        batch_idxs = [node_mapping[c] for c in batch_candidates]
        
        # Get source embeddings (repeated for each candidate)
        src_embeddings = paper_features_tensor.repeat(len(batch_candidates), 1)
        
        # Get target embeddings
        tgt_embeddings = node_features[torch.tensor(batch_idxs, dtype=torch.long).to(device)]
        
        # Extract edge features for each candidate
        edge_features = []
        for candidate in batch_candidates:
            features = extract_edge_features(
                paper_id, 
                candidate, 
                G, 
                metadata, 
                author_coauthor_G,
                observed_citations_only=True,
                observed_citations=observed_citations
            )
            edge_features.append(list(features.values()))
        
        # Convert edge features to tensor
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(src_embeddings, tgt_embeddings, edge_features_tensor)
            probs = outputs.squeeze().cpu().numpy()
        
        all_probs.extend(probs)
    
    # Rank candidates by probability
    ranked_indices = np.argsort(-np.array(all_probs))
    ranked_candidates = [candidates[i] for i in ranked_indices]
    ranked_probs = [all_probs[i] for i in ranked_indices]
    
    # Get top-k recommendations
    recommendations = []
    for i, (candidate, prob) in enumerate(zip(ranked_candidates[:top_k], ranked_probs[:top_k])):
        # Get metadata for the candidate
        candidate_metadata = metadata.get(candidate, {})
        recommendation = {
            'rank': i + 1,
            'id': candidate,
            'title': candidate_metadata.get('title', ''),
            'authors': candidate_metadata.get('authors', ''),
            'journal': candidate_metadata.get('journal', ''),
            'score': float(prob)
        }
        recommendations.append(recommendation)
    
    return recommendations, paper_id, observed_citations, paper_metadata

def inductive_evaluate_with_held_out(paper_json, G, metadata, model, node_features, 
                                   node_mapping, reverse_mapping, author_coauthor_G=None, 
                                   observed_ratio=0.75, top_k=10, device='cpu'):
    """
    Evaluate citation recommendations for a new paper with held-out citations.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph with existing papers
        metadata: Dictionary of paper metadata
        model: Trained EnhancedCitationMLP model
        node_features: Node features tensor for existing papers
        node_mapping: Mapping from node IDs to indices
        reverse_mapping: Mapping from indices to node IDs
        author_coauthor_G: Author co-authorship graph (optional)
        observed_ratio: Ratio of citations to observe
        top_k: Number of top recommendations to consider
        device: Device to run the model on
        
    Returns:
        Dictionary of evaluation metrics and recommendations
    """
    # Extract all citations
    all_citations = []
    for ref in paper_json.get('references', []):
        if ref.get('reference_type') != 'article':
            continue
            
        ref_title = ref.get('title')
        if not ref_title or should_filter_title(ref_title):
            continue
            
        ref_id = get_paper_id(
            ref_title,
            ref.get('authors', []),
            str(ref.get('year', ''))
        )
        
        if ref_id and ref_id in G:
            all_citations.append(ref_id)
    
    if len(all_citations) < 2:
        return {'error': 'Paper has too few citations for evaluation'}
    
    # Split citations into observed and held-out sets
    random.shuffle(all_citations)
    num_observed = max(1, int(len(all_citations) * observed_ratio))
    observed_citations = all_citations[:num_observed]
    held_out_citations = all_citations[num_observed:]
    
    # Create a modified paper_json with only observed citations
    modified_paper_json = paper_json.copy()
    modified_references = []
    
    for ref in paper_json.get('references', []):
        ref_id = get_paper_id(
            ref.get('title', ''),
            ref.get('authors', []),
            str(ref.get('year', ''))
        )
        
        if ref_id in observed_citations:
            modified_references.append(ref)
    
    modified_paper_json['references'] = modified_references
    
    # Get recommendations using the inductive approach
    recommendations, paper_id, _, paper_metadata = inductive_recommend_citations(
        modified_paper_json, 
        G, 
        metadata, 
        model, 
        node_features, 
        node_mapping, 
        reverse_mapping, 
        author_coauthor_G,
        top_k=top_k,
        exclude_observed=True,
        device=device
    )
    
    # Calculate metrics
    recommended_ids = [rec['id'] for rec in recommendations]
    hits = [rec for rec in recommendations if rec['id'] in held_out_citations]
    
    # Recall@k: proportion of held-out citations in top k recommendations
    recall = len(hits) / len(held_out_citations) if held_out_citations else 0
    
    # Precision@k: proportion of top k recommendations that are correct
    precision = len(hits) / len(recommendations) if recommendations else 0
    
    # NDCG@k
    dcg = 0
    for i, rec in enumerate(recommendations):
        if rec['id'] in held_out_citations:
            # Log base 2 ranking (starting from 1)
            dcg += 1 / np.log2(i + 2)
    
    # Ideal DCG: all held-out citations at the top positions
    idcg = 0
    for i in range(min(len(held_out_citations), top_k)):
        idcg += 1 / np.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0
    
    # Return metrics and recommendations
    return {
        'paper_id': paper_id,
        'paper_title': paper_metadata['title'],
        'total_citations': len(all_citations),
        'observed_citations': len(observed_citations),
        'held_out_citations': len(held_out_citations),
        'hits': len(hits),
        f'recall@{top_k}': recall,
        f'precision@{top_k}': precision,
        f'ndcg@{top_k}': ndcg,
        'recommendations': recommendations,
        'held_out_citation_ids': held_out_citations
    }

def load_articles(articles_file='articles.jsonl'):
    """
    Load articles from a JSONL file or from cache.
    
    Args:
        articles_file: Path to the articles file
        
    Returns:
        List of article dictionaries
    """
    # Create cache directory if it doesn't exist
    os.makedirs('cache_gnn', exist_ok=True)
    
    # Load articles
    articles_cache_path = 'cache_gnn/articles.pkl'
    if os.path.exists(articles_cache_path):
        print("Loading articles from cache...")
        with open(articles_cache_path, 'rb') as f:
            articles = pkl.load(f)
    else:
        print(f"Loading articles from {articles_file}...")
        with open(articles_file, 'r') as f:
            articles = [json.loads(line) for line in f]
        # Cache articles
        with open(articles_cache_path, 'wb') as f:
            pkl.dump(articles, f)
    
    print(f"Loaded {len(articles)} articles")
    return articles

def build_gnn_graph(articles, embedding_file='embeddings/all_papers_with_embeddings.pkl'):
    """
    Build a graph for GNN-based citation recommendation.
    
    Args:
        articles: List of article dictionaries
        embedding_file: Path to the file containing paper embeddings
        
    Returns:
        Tuple of (graph, metadata_dict, main_article_count)
    """
    # Check if graph already exists in cache
    gnn_graph_cache_path = 'cache_gnn/gnn_citation_graph.gpickle'
    gnn_metadata_cache_path = 'cache_gnn/gnn_paper_metadata.json'
    
    if os.path.exists(gnn_graph_cache_path) and os.path.exists(gnn_metadata_cache_path):
        print("Loading GNN graph and metadata from cache...")
        with open(gnn_graph_cache_path, 'rb') as f:
            G = pkl.load(f)
        with open(gnn_metadata_cache_path, 'r') as f:
            metadata = json.load(f)
        main_article_count = len([node for node in metadata if metadata[node].get('is_main_article', False)])
    else:
        print("Creating GNN graph from articles...")
        G, metadata, main_article_count = process_articles_to_gnn_graph(articles, embedding_file)
        # Save graph and metadata
        save_gnn_graph(G, metadata)
    
    # Print graph statistics
    print_graph_statistics(G, metadata)
    
    return G, metadata, main_article_count

def convert_to_pytorch_geometric(G, metadata):
    """
    Convert NetworkX graph to PyTorch Geometric format.
    
    Args:
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        
    Returns:
        PyTorch Geometric Data object
    """
    # Extract node features
    node_ids = list(G.nodes())
    node_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # Create feature matrices
    num_nodes = len(node_ids)
    
    # Get embedding dimension from first node
    embedding_dim = len(G.nodes[node_ids[0]]['scibert_embedding'])
    
    # Initialize feature matrices
    year_features = np.zeros((num_nodes, 1))
    journal_features = np.zeros((num_nodes, 8))  # Assuming 8 journals
    embedding_features = np.zeros((num_nodes, embedding_dim))
    is_main_article = np.zeros((num_nodes, 1))
    
    # Fill feature matrices
    for i, node_id in enumerate(node_ids):
        # Year feature (normalized)
        year = G.nodes[node_id].get('year', 0)
        if isinstance(year, str) and year.isdigit():
            year = int(year)
        year_features[i, 0] = year
        
        # Journal encoding
        journal_encoding = G.nodes[node_id].get('journal_encoding', [0] * 8)
        journal_features[i] = journal_encoding
        
        # SciBERT embedding
        embedding = G.nodes[node_id].get('scibert_embedding', [0] * embedding_dim)
        embedding_features[i] = embedding
        
        # Is main article
        is_main_article[i, 0] = 1 if metadata[node_id].get('is_main_article', False) else 0
    
    # Normalize year features
    if np.std(year_features) > 0:
        year_features = (year_features - np.mean(year_features)) / np.std(year_features)
    
    # Combine features
    node_features = np.concatenate([
        year_features,
        journal_features,
        embedding_features,
        is_main_article
    ], axis=1)
    
    # Convert to PyTorch tensors
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge index
    edges = list(G.edges())
    edge_index = torch.tensor([[node_mapping[src], node_mapping[dst]] for src, dst in edges], 
                             dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=node_features_tensor, edge_index=edge_index)
    
    # Store node mapping for reference
    data.node_mapping = node_mapping
    data.reverse_mapping = {i: node_id for node_id, i in node_mapping.items()}
    
    print(f"Created PyTorch Geometric Data object with {data.num_nodes} nodes and {data.num_edges} edges")
    print(f"Node feature dimensions: {data.num_node_features}")
    
    return data

def prepare_citation_prediction_data_with_edge_features(G, metadata, author_coauthor_G=None, split_ratio=0.8, random_seed=42):
    """
    Prepare data for citation prediction task with edge features.
    
    Args:
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        author_coauthor_G: Author co-authorship graph (optional)
        split_ratio: Ratio of training to test data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing training and test data with edge features
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get main articles
    main_articles = [node for node in G.nodes() if metadata[node].get('is_main_article', True)]
    print(f"Found {len(main_articles)} main articles")
    
    # Create positive and negative examples with edge features
    positive_examples = []
    negative_examples = []
    
    for main_article in tqdm(main_articles, desc="Creating citation examples with edge features"):
        # Get papers that the main article cites
        cited_papers = list(G.successors(main_article))
        
        # Add all positive examples with edge features
        for cited_paper in cited_papers:
            # Extract edge features
            edge_features = extract_edge_features(main_article, cited_paper, G, metadata, author_coauthor_G)
            positive_examples.append((main_article, cited_paper, 1, edge_features))
        
        # Get all papers that could be cited but aren't
        all_papers = list(G.nodes())
        non_cited_papers = [p for p in all_papers if p != main_article and p not in cited_papers]
        
        # Sample negative examples (5x the number of positive examples)
        num_negative_samples = min(len(non_cited_papers), 5 * len(cited_papers))
        if len(non_cited_papers) > num_negative_samples:
            non_cited_papers = random.sample(non_cited_papers, num_negative_samples)
        
        # Add negative examples with edge features
        for non_cited_paper in non_cited_papers:
            # Extract edge features
            edge_features = extract_edge_features(main_article, non_cited_paper, G, metadata, author_coauthor_G)
            negative_examples.append((main_article, non_cited_paper, 0, edge_features))
    
    print(f"Created {len(positive_examples)} positive examples and {len(negative_examples)} negative examples")
    
    # Combine and shuffle examples
    all_examples = positive_examples + negative_examples
    random.shuffle(all_examples)
    
    # Split into training and test sets
    split_idx = int(len(all_examples) * split_ratio)
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]
    
    # Create node mapping for PyTorch Geometric
    node_ids = list(G.nodes())
    node_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # Convert examples to tensor format
    train_source_nodes = [node_mapping[src] for src, _, _, _ in train_examples]
    train_target_nodes = [node_mapping[dst] for _, dst, _, _ in train_examples]
    train_labels = [label for _, _, label, _ in train_examples]
    train_edge_features = [list(features.values()) for _, _, _, features in train_examples]
    
    test_source_nodes = [node_mapping[src] for src, _, _, _ in test_examples]
    test_target_nodes = [node_mapping[dst] for _, dst, _, _ in test_examples]
    test_labels = [label for _, _, label, _ in test_examples]
    test_edge_features = [list(features.values()) for _, _, _, features in test_examples]
    
    # Convert to PyTorch tensors
    train_source_tensor = torch.tensor(train_source_nodes, dtype=torch.long)
    train_target_tensor = torch.tensor(train_target_nodes, dtype=torch.long)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)
    train_edge_features_tensor = torch.tensor(train_edge_features, dtype=torch.float)
    
    test_source_tensor = torch.tensor(test_source_nodes, dtype=torch.long)
    test_target_tensor = torch.tensor(test_target_nodes, dtype=torch.long)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float)
    test_edge_features_tensor = torch.tensor(test_edge_features, dtype=torch.float)
    
    # Create data dictionary
    data_dict = {
        'train': {
            'source_nodes': train_source_tensor,
            'target_nodes': train_target_tensor,
            'edge_features': train_edge_features_tensor,
            'labels': train_labels_tensor,
            'examples': train_examples
        },
        'test': {
            'source_nodes': test_source_tensor,
            'target_nodes': test_target_tensor,
            'edge_features': test_edge_features_tensor,
            'labels': test_labels_tensor,
            'examples': test_examples
        },
        'node_mapping': node_mapping,
        'reverse_mapping': {i: node_id for node_id, i in node_mapping.items()},
        'edge_feature_names': list(train_examples[0][3].keys()) if train_examples else []
    }
    
    print(f"Created training set with {len(train_examples)} examples")
    print(f"Created test set with {len(test_examples)} examples")
    print(f"Edge features: {data_dict['edge_feature_names']}")
    
    return data_dict

def extract_edge_features(source_id, target_id, G, metadata, author_coauthor_G=None, observed_citations_only=False, observed_citations=None):
    """
    Extract edge features for a pair of papers.
    
    Args:
        source_id: ID of the source paper
        target_id: ID of the target paper
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        author_coauthor_G: Author co-authorship graph (optional)
        observed_citations_only: Whether to use only observed citations for feature extraction
        observed_citations: List of observed citation IDs (used when observed_citations_only=True)
        
    Returns:
        Dictionary of edge features
    """
    features = {}
    
    # Get metadata for both papers
    source_metadata = metadata.get(source_id, {})
    target_metadata = metadata.get(target_id, {})
    
    # 1. Number of shared authors
    authors1 = set(source_metadata.get('authors', '').split('; '))
    authors2 = set(target_metadata.get('authors', '').split('; '))
    features['shared_authors'] = len(authors1.intersection(authors2))
    
    # 2. Number of shared citations
    if observed_citations_only and observed_citations is not None:
        # Use only observed citations for feature extraction
        citations1 = set(observed_citations)
    else:
        # Use all citations
        citations1 = set(G.successors(source_id)) if source_id in G else set()
    
    citations2 = set(G.successors(target_id)) if target_id in G else set()
    features['shared_citations'] = len(citations1.intersection(citations2))
    
    # 3. Common coauthor weight (if author_coauthor_G is provided)
    common_coauthor_weight = 0
    
    if author_coauthor_G is not None:
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
    
    features['common_coauthor_weight'] = common_coauthor_weight
    
    # 4. Add new features for partial citation prediction
    if observed_citations_only and observed_citations is not None:
        # Calculate how many observed citations cite the target paper
        target_citations = set(G.predecessors(target_id)) if target_id in G else set()
        observed_citing_target = len([c for c in observed_citations if c in target_citations])
        features['observed_citing_target'] = observed_citing_target
        
        # Calculate how many papers are cited by both observed citations and the target
        target_cites = set(G.successors(target_id)) if target_id in G else set()
        observed_citation_cites = set()
        for c in observed_citations:
            observed_citation_cites.update(G.successors(c) if c in G else set())
        features['common_citations_with_observed'] = len(observed_citation_cites.intersection(target_cites))
    else:
        features['observed_citing_target'] = 0
        features['common_citations_with_observed'] = 0
    
    return features

def process_articles_to_author_graphs(articles):
    """
    Create author coauthorship graph from articles.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        NetworkX graph of author coauthorships
    """
    # Create undirected graph for coauthorships
    author_coauthor_G = nx.Graph()
    
    # Dictionary to track coauthorship counts
    coauthor_counts = defaultdict(lambda: defaultdict(int))
    
    # Process each article
    print("Building author coauthorship graph...")
    for article in tqdm(articles):
        # Skip if missing key fields
        if not article.get('title') or not article.get('published_date') or not article.get('authors'):
            continue
            
        # Skip articles that should be filtered out
        if should_filter_title(article.get('title', '')):
            continue
            
        # Format authors
        formatted_authors = []
        for author in article.get('authors', []):
            formatted_name = format_author_name(author)
            if formatted_name:
                formatted_authors.append(formatted_name)
        
        # Process coauthorship relationships
        for i, author1 in enumerate(formatted_authors):
            for author2 in formatted_authors[i+1:]:
                coauthor_counts[author1][author2] += 1
                coauthor_counts[author2][author1] += 1
    
    # Build the coauthorship graph
    for author1, coauthors in coauthor_counts.items():
        for author2, coauthor_weight in coauthors.items():
            author_coauthor_G.add_edge(author1, author2, weight=coauthor_weight)
    
    print(f"Created author coauthorship graph with {len(author_coauthor_G.nodes())} authors and {len(author_coauthor_G.edges())} coauthorship relationships")
    
    return author_coauthor_G

def test_edge_features():
    """
    Test function to extract edge features for a sample of paper pairs.
    """
    print("Testing edge feature extraction...")
    
    # Load articles
    articles = load_articles()
    
    # Build GNN graph
    G, metadata, _ = build_gnn_graph(articles)
    
    # Build author coauthorship graph
    author_coauthor_G = process_articles_to_author_graphs(articles)
    
    # Get a sample of main articles
    main_articles = [node for node in G.nodes() if metadata[node].get('is_main_article', True)]
    sample_main_articles = random.sample(main_articles, min(5, len(main_articles)))
    
    # For each sample main article, get a cited paper and a non-cited paper
    for main_article in sample_main_articles:
        print(f"\nMain article: {metadata[main_article].get('title')}")
        
        # Get a cited paper
        cited_papers = list(G.successors(main_article))
        if cited_papers:
            cited_paper = random.choice(cited_papers)
            print(f"Cited paper: {metadata[cited_paper].get('title')}")
            
            # Extract edge features
            features = extract_edge_features(main_article, cited_paper, G, metadata, author_coauthor_G)
            print("Edge features for cited paper:")
            for feature, value in features.items():
                print(f"  {feature}: {value}")
        
        # Get a non-cited paper
        all_papers = list(G.nodes())
        non_cited_papers = [p for p in all_papers if p != main_article and p not in G.successors(main_article)]
        if non_cited_papers:
            non_cited_paper = random.choice(non_cited_papers)
            print(f"Non-cited paper: {metadata[non_cited_paper].get('title')}")
            
            # Extract edge features
            features = extract_edge_features(main_article, non_cited_paper, G, metadata, author_coauthor_G)
            print("Edge features for non-cited paper:")
            for feature, value in features.items():
                print(f"  {feature}: {value}")

class EnhancedCitationMLP(nn.Module):
    """
    Enhanced Multilayer Perceptron for citation prediction with partial citation information.
    
    Combines node embeddings, edge features, and citation context to predict missing citations.
    """
    def __init__(self, node_embedding_dim, edge_feature_dim, hidden_dims=[128, 64]):
        super(EnhancedCitationMLP, self).__init__()
        
        # Input dimensions
        self.node_embedding_dim = node_embedding_dim
        self.edge_feature_dim = edge_feature_dim
        self.input_dim = 2 * node_embedding_dim + edge_feature_dim  # Source + target embeddings + edge features
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, source_embeddings, target_embeddings, edge_features):
        """
        Forward pass through the MLP.
        
        Args:
            source_embeddings: Embeddings of source nodes [batch_size, node_embedding_dim]
            target_embeddings: Embeddings of target nodes [batch_size, node_embedding_dim]
            edge_features: Edge features [batch_size, edge_feature_dim]
            
        Returns:
            Citation prediction probabilities [batch_size, 1]
        """
        # Concatenate source embeddings, target embeddings, and edge features
        combined_features = torch.cat([source_embeddings, target_embeddings, edge_features], dim=1)
        
        # Pass through MLP
        return self.mlp(combined_features)

def train_citation_mlp(data, citation_data, device='cuda', batch_size=64, epochs=10, lr=0.001):
    """
    Train a MLP model for citation prediction.
    
    Args:
        data: PyTorch Geometric Data object with node features
        citation_data: Dictionary with citation prediction data
        device: Device to train on ('cuda' or 'cpu')
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    # Get node embeddings
    node_features = data.x.to(device)
    
    # Get training data
    train_source_nodes = citation_data['train']['source_nodes'].to(device)
    train_target_nodes = citation_data['train']['target_nodes'].to(device)
    train_edge_features = citation_data['train']['edge_features'].to(device)
    train_labels = citation_data['train']['labels'].to(device)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(train_source_nodes, train_target_nodes, train_edge_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Get dimensions
    node_embedding_dim = node_features.shape[1]
    edge_feature_dim = train_edge_features.shape[1]
    
    # Initialize model
    model = EnhancedCitationMLP(node_embedding_dim, edge_feature_dim).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_src, batch_tgt, batch_edge, batch_labels in progress_bar:
            # Get node embeddings for source and target nodes
            src_embeddings = node_features[batch_src]
            tgt_embeddings = node_features[batch_tgt]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(src_embeddings, tgt_embeddings, batch_edge)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Evaluate on validation set
        val_metrics = evaluate_citation_mlp(model, data, citation_data, device, batch_size)
        
        # Update history
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_metrics['loss']:.4f} - "
              f"Val Accuracy: {val_metrics['accuracy']:.4f} - Val F1: {val_metrics['f1']:.4f} - "
              f"Val AUC: {val_metrics['auc']:.4f}")
    
    return model, history

def evaluate_citation_mlp(model, data, citation_data, device='cuda', batch_size=64):
    """
    Evaluate the MLP model on the test set.
    
    Args:
        model: Trained CitationMLP model
        data: PyTorch Geometric Data object with node features
        citation_data: Dictionary with citation prediction data
        device: Device to evaluate on ('cuda' or 'cpu')
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Get node embeddings
    node_features = data.x.to(device)
    
    # Get test data
    test_source_nodes = citation_data['test']['source_nodes'].to(device)
    test_target_nodes = citation_data['test']['target_nodes'].to(device)
    test_edge_features = citation_data['test']['edge_features'].to(device)
    test_labels = citation_data['test']['labels'].to(device)
    
    # Create dataset and dataloader
    test_dataset = TensorDataset(test_source_nodes, test_target_nodes, test_edge_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Evaluation
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_src, batch_tgt, batch_edge, batch_labels in test_loader:
            # Get node embeddings for source and target nodes
            src_embeddings = node_features[batch_src]
            tgt_embeddings = node_features[batch_tgt]
            
            # Forward pass
            outputs = model(src_embeddings, tgt_embeddings, batch_edge)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Store predictions and labels
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels = batch_labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Return metrics
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Dictionary of training history
        
    Returns:
        None
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axs[0, 0].plot(history['loss'], label='Training Loss')
    axs[0, 0].plot(history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Plot accuracy
    axs[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    
    # Plot precision, recall, F1
    axs[1, 0].plot(history['val_precision'], label='Precision')
    axs[1, 0].plot(history['val_recall'], label='Recall')
    axs[1, 0].plot(history['val_f1'], label='F1 Score')
    axs[1, 0].set_title('Precision, Recall, F1')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].legend()
    
    # Plot AUC
    axs[1, 1].plot(history['val_auc'], label='AUC')
    axs[1, 1].set_title('AUC')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('AUC')
    axs[1, 1].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('citation_mlp_training_history.png')
    plt.close()

def analyze_feature_importance(model, data, citation_data, device='cuda'):
    """
    Analyze feature importance for the citation prediction model.
    
    Args:
        model: Trained model (CitationMLP or EnhancedCitationMLP)
        data: PyTorch Geometric Data object with node features
        citation_data: Dictionary with citation prediction data
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    model.eval()
    
    # Get node embeddings
    node_features = data.x.to(device)
    
    # Get test data
    test_source_nodes = citation_data['test']['source_nodes'].to(device)
    test_target_nodes = citation_data['test']['target_nodes'].to(device)
    test_edge_features = citation_data['test']['edge_features'].to(device)
    test_labels = citation_data['test']['labels'].to(device)
    
    # Get feature names
    feature_names = citation_data['edge_feature_names']
    
    # Create a baseline prediction
    with torch.no_grad():
        src_embeddings = node_features[test_source_nodes]
        tgt_embeddings = node_features[test_target_nodes]
        baseline_outputs = model(src_embeddings, tgt_embeddings, test_edge_features)
        baseline_probs = baseline_outputs.squeeze().cpu().numpy()
    
    # Calculate importance for each feature
    feature_importance = {}
    
    for i, feature_name in enumerate(feature_names):
        # Create a copy of edge features with the current feature zeroed out
        modified_edge_features = test_edge_features.clone()
        modified_edge_features[:, i] = 0
        
        # Get predictions with the modified features
        with torch.no_grad():
            modified_outputs = model(src_embeddings, tgt_embeddings, modified_edge_features)
            modified_probs = modified_outputs.squeeze().cpu().numpy()
        
        # Calculate the absolute difference in predictions
        importance = np.mean(np.abs(baseline_probs - modified_probs))
        feature_importance[feature_name] = float(importance)
    
    # Normalize importance scores
    total_importance = sum(feature_importance.values())
    if total_importance > 0:
        for feature_name in feature_importance:
            feature_importance[feature_name] /= total_importance
    
    # Sort by importance
    feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
    
    return feature_importance

def plot_confusion_matrix(y_true, y_pred, classes=['No Citation', 'Citation'], normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Plot confusion matrix for citation prediction results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Color map
        
    Returns:
        None
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig('citation_confusion_matrix.png')
    plt.close()
    
    return ax

def prepare_partial_citation_prediction_data(G, metadata, author_coauthor_G=None, observed_ratio=0.75, split_ratio=0.8, random_seed=42):
    """
    Prepare data for partial citation prediction task with edge features.
    For each main article, we observe only a portion of its citations and predict the rest.
    
    Args:
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        author_coauthor_G: Author co-authorship graph (optional)
        observed_ratio: Ratio of citations to observe for each paper (e.g., 0.75 means we observe 75% of citations)
        split_ratio: Ratio of papers to use for training vs. testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing training and test data with edge features
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get main articles
    main_articles = [node for node in G.nodes() if metadata[node].get('is_main_article', True)]
    print(f"Found {len(main_articles)} main articles")
    
    # Split main articles into training and test sets
    random.shuffle(main_articles)
    split_idx = int(len(main_articles) * split_ratio)
    train_main_articles = main_articles[:split_idx]
    test_main_articles = main_articles[split_idx:]
    
    print(f"Using {len(train_main_articles)} main articles for training")
    print(f"Using {len(test_main_articles)} main articles for testing")
    
    # Create examples for training and testing
    train_examples = []
    test_examples = []
    
    # Process training articles
    print("Creating training examples with partial citation information...")
    for main_article in tqdm(train_main_articles, desc="Processing training articles"):
        # Get papers that the main article cites
        cited_papers = list(G.successors(main_article))
        
        if len(cited_papers) < 2:
            # Skip articles with too few citations
            continue
        
        # Randomly split citations into observed and held-out sets
        random.shuffle(cited_papers)
        num_observed = max(1, int(len(cited_papers) * observed_ratio))
        observed_citations = cited_papers[:num_observed]
        held_out_citations = cited_papers[num_observed:]
        
        # Add positive examples for held-out citations
        for cited_paper in held_out_citations:
            # Extract edge features (using only observed citations for feature extraction)
            edge_features = extract_edge_features(
                main_article, 
                cited_paper, 
                G, 
                metadata, 
                author_coauthor_G,
                observed_citations_only=True,
                observed_citations=observed_citations
            )
            train_examples.append((main_article, cited_paper, 1, edge_features, observed_citations))
        
        # Sample negative examples (papers that are not cited)
        all_papers = list(G.nodes())
        non_cited_papers = [p for p in all_papers if p != main_article and p not in cited_papers]
        
        # Sample negative examples (same number as positive examples for balance)
        num_negative_samples = min(len(non_cited_papers), len(held_out_citations) * 5)  # 5:1 ratio
        if len(non_cited_papers) > num_negative_samples:
            non_cited_papers = random.sample(non_cited_papers, num_negative_samples)
        
        # Add negative examples
        for non_cited_paper in non_cited_papers:
            # Extract edge features (using only observed citations)
            edge_features = extract_edge_features(
                main_article, 
                non_cited_paper, 
                G, 
                metadata, 
                author_coauthor_G,
                observed_citations_only=True,
                observed_citations=observed_citations
            )
            train_examples.append((main_article, non_cited_paper, 0, edge_features, observed_citations))
    
    # Process test articles
    print("Creating test examples with partial citation information...")
    for main_article in tqdm(test_main_articles, desc="Processing test articles"):
        # Get papers that the main article cites
        cited_papers = list(G.successors(main_article))
        
        if len(cited_papers) < 2:
            # Skip articles with too few citations
            continue
        
        # Randomly split citations into observed and held-out sets
        random.shuffle(cited_papers)
        num_observed = max(1, int(len(cited_papers) * observed_ratio))
        observed_citations = cited_papers[:num_observed]
        held_out_citations = cited_papers[num_observed:]
        
        # Add positive examples for held-out citations
        for cited_paper in held_out_citations:
            # Extract edge features (using only observed citations)
            edge_features = extract_edge_features(
                main_article, 
                cited_paper, 
                G, 
                metadata, 
                author_coauthor_G,
                observed_citations_only=True,
                observed_citations=observed_citations
            )
            test_examples.append((main_article, cited_paper, 1, edge_features, observed_citations))
        
        # Sample negative examples (papers that are not cited)
        all_papers = list(G.nodes())
        non_cited_papers = [p for p in all_papers if p != main_article and p not in cited_papers]
        
        # Sample negative examples (same number as positive examples for balance)
        num_negative_samples = min(len(non_cited_papers), len(held_out_citations) * 5)  # 5:1 ratio
        if len(non_cited_papers) > num_negative_samples:
            non_cited_papers = random.sample(non_cited_papers, num_negative_samples)
        
        # Add negative examples
        for non_cited_paper in non_cited_papers:
            # Extract edge features (using only observed citations)
            edge_features = extract_edge_features(
                main_article, 
                non_cited_paper, 
                G, 
                metadata, 
                author_coauthor_G,
                observed_citations_only=True,
                observed_citations=observed_citations
            )
            test_examples.append((main_article, non_cited_paper, 0, edge_features, observed_citations))
    
    print(f"Created {len(train_examples)} training examples")
    print(f"Created {len(test_examples)} test examples")
    
    # Create node mapping for PyTorch Geometric
    node_ids = list(G.nodes())
    node_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # Convert examples to tensor format
    train_source_nodes = [node_mapping[src] for src, _, _, _, _ in train_examples]
    train_target_nodes = [node_mapping[dst] for _, dst, _, _, _ in train_examples]
    train_labels = [label for _, _, label, _, _ in train_examples]
    train_edge_features = [list(features.values()) for _, _, _, features, _ in train_examples]
    train_observed_citations = [[node_mapping[c] for c in obs_cites] for _, _, _, _, obs_cites in train_examples]
    
    test_source_nodes = [node_mapping[src] for src, _, _, _, _ in test_examples]
    test_target_nodes = [node_mapping[dst] for _, dst, _, _, _ in test_examples]
    test_labels = [label for _, _, label, _, _ in test_examples]
    test_edge_features = [list(features.values()) for _, _, _, features, _ in test_examples]
    test_observed_citations = [[node_mapping[c] for c in obs_cites] for _, _, _, _, obs_cites in test_examples]
    
    # Convert to PyTorch tensors
    train_source_tensor = torch.tensor(train_source_nodes, dtype=torch.long)
    train_target_tensor = torch.tensor(train_target_nodes, dtype=torch.long)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float)
    train_edge_features_tensor = torch.tensor(train_edge_features, dtype=torch.float)
    
    test_source_tensor = torch.tensor(test_source_nodes, dtype=torch.long)
    test_target_tensor = torch.tensor(test_target_nodes, dtype=torch.long)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float)
    test_edge_features_tensor = torch.tensor(test_edge_features, dtype=torch.float)
    
    # Create data dictionary
    data_dict = {
        'train': {
            'source_nodes': train_source_tensor,
            'target_nodes': train_target_tensor,
            'edge_features': train_edge_features_tensor,
            'labels': train_labels_tensor,
            'observed_citations': train_observed_citations,
            'examples': train_examples
        },
        'test': {
            'source_nodes': test_source_tensor,
            'target_nodes': test_target_tensor,
            'edge_features': test_edge_features_tensor,
            'labels': test_labels_tensor,
            'observed_citations': test_observed_citations,
            'examples': test_examples
        },
        'node_mapping': node_mapping,
        'reverse_mapping': {i: node_id for node_id, i in node_mapping.items()},
        'edge_feature_names': list(train_examples[0][3].keys()) if train_examples else []
    }
    
    return data_dict

def train_partial_citation_mlp(data, citation_data, device='cuda', batch_size=64, epochs=10, lr=0.001):
    """
    Train a MLP model for partial citation prediction.
    
    Args:
        data: PyTorch Geometric Data object with node features
        citation_data: Dictionary with partial citation prediction data
        device: Device to train on ('cuda' or 'cpu')
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    # Get node embeddings
    node_features = data.x.to(device)
    
    # Get training data
    train_source_nodes = citation_data['train']['source_nodes'].to(device)
    train_target_nodes = citation_data['train']['target_nodes'].to(device)
    train_edge_features = citation_data['train']['edge_features'].to(device)
    train_labels = citation_data['train']['labels'].to(device)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(train_source_nodes, train_target_nodes, train_edge_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Get dimensions
    node_embedding_dim = node_features.shape[1]
    edge_feature_dim = train_edge_features.shape[1]
    
    # Initialize model
    model = EnhancedCitationMLP(node_embedding_dim, edge_feature_dim).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_src, batch_tgt, batch_edge, batch_labels in progress_bar:
            # Get node embeddings for source and target nodes
            src_embeddings = node_features[batch_src]
            tgt_embeddings = node_features[batch_tgt]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(src_embeddings, tgt_embeddings, batch_edge)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        # Evaluate on validation set
        val_metrics = evaluate_partial_citation_mlp(model, data, citation_data, device, batch_size)
        
        # Update history
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_metrics['loss']:.4f} - "
              f"Val Accuracy: {val_metrics['accuracy']:.4f} - Val F1: {val_metrics['f1']:.4f} - "
              f"Val AUC: {val_metrics['auc']:.4f}")
    
    return model, history

def evaluate_partial_citation_mlp(model, data, citation_data, device='cuda', batch_size=64):
    """
    Evaluate the MLP model on the test set for partial citation prediction.
    
    Args:
        model: Trained EnhancedCitationMLP model
        data: PyTorch Geometric Data object with node features
        citation_data: Dictionary with partial citation prediction data
        device: Device to evaluate on ('cuda' or 'cpu')
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Get node embeddings
    node_features = data.x.to(device)
    
    # Get test data
    test_source_nodes = citation_data['test']['source_nodes'].to(device)
    test_target_nodes = citation_data['test']['target_nodes'].to(device)
    test_edge_features = citation_data['test']['edge_features'].to(device)
    test_labels = citation_data['test']['labels'].to(device)
    
    # Create dataset and dataloader
    test_dataset = TensorDataset(test_source_nodes, test_target_nodes, test_edge_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Evaluation
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_src, batch_tgt, batch_edge, batch_labels in test_loader:
            # Get node embeddings for source and target nodes
            src_embeddings = node_features[batch_src]
            tgt_embeddings = node_features[batch_tgt]
            
            # Forward pass
            outputs = model(src_embeddings, tgt_embeddings, batch_edge)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Store predictions and labels
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            labels = batch_labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Return metrics
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def evaluate_citation_recommendations(model, data, citation_data, G, metadata, top_k=10, device='cuda'):
    """
    Evaluate citation recommendations for papers with partial citation information.
    
    Args:
        model: Trained EnhancedCitationMLP model
        data: PyTorch Geometric Data object with node features
        citation_data: Dictionary with partial citation prediction data
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        top_k: Number of top recommendations to consider
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        Dictionary of recommendation metrics
    """
    model.eval()
    node_mapping = citation_data['node_mapping']
    reverse_mapping = citation_data['reverse_mapping']
    
    # Get node embeddings
    node_features = data.x.to(device)
    
    # Group test examples by source paper
    paper_examples = {}
    for i, (src, dst, label, _, observed) in enumerate(citation_data['test']['examples']):
        if src not in paper_examples:
            paper_examples[src] = {
                'observed_citations': observed,
                'held_out_citations': [],
                'candidates': [],
                'candidate_indices': []
            }
        
        if label == 1:
            # This is a held-out citation
            paper_examples[src]['held_out_citations'].append(dst)
        
        # Add to candidates (both positive and negative examples)
        paper_examples[src]['candidates'].append(dst)
        paper_examples[src]['candidate_indices'].append(i)
    
    # Metrics
    recall_at_k = []
    precision_at_k = []
    ndcg_at_k = []
    
    # Process each paper
    for src_id, paper_data in tqdm(paper_examples.items(), desc="Evaluating recommendations"):
        # Skip papers with no held-out citations
        if not paper_data['held_out_citations']:
            continue
        
        # Get candidate indices
        candidate_indices = paper_data['candidate_indices']
        candidates = paper_data['candidates']
        held_out_citations = set(paper_data['held_out_citations'])
        
        # Skip if no candidates
        if not candidate_indices:
            continue
        
        # Get source node index
        src_idx = node_mapping[src_id]
        
        # Get predictions for all candidates
        candidate_idxs = [node_mapping[c] for c in candidates]
        src_embeddings = node_features[src_idx].repeat(len(candidate_idxs), 1)
        tgt_embeddings = node_features[torch.tensor(candidate_idxs, dtype=torch.long).to(device)]
        
        # Get edge features for candidates
        edge_features = citation_data['test']['edge_features'][torch.tensor(candidate_indices)].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(src_embeddings, tgt_embeddings, edge_features)
            probs = outputs.squeeze().cpu().numpy()
        
        # Rank candidates by probability
        ranked_indices = np.argsort(-probs)
        ranked_candidates = [candidates[i] for i in ranked_indices]
        
        # Calculate metrics
        # Recall@k: proportion of held-out citations in top k recommendations
        top_k_candidates = ranked_candidates[:top_k]
        hits = sum(1 for c in top_k_candidates if c in held_out_citations)
        recall = hits / len(held_out_citations)
        recall_at_k.append(recall)
        
        # Precision@k: proportion of top k recommendations that are correct
        precision = hits / min(top_k, len(top_k_candidates))
        precision_at_k.append(precision)
        
        # NDCG@k: normalized discounted cumulative gain
        dcg = 0
        idcg = 0
        for i, candidate in enumerate(top_k_candidates):
            if candidate in held_out_citations:
                # Log base 2 ranking (starting from 1)
                dcg += 1 / np.log2(i + 2)
        
        # Ideal DCG: all held-out citations at the top positions
        for i in range(min(len(held_out_citations), top_k)):
            idcg += 1 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_at_k.append(ndcg)
    
    # Calculate average metrics
    avg_recall = np.mean(recall_at_k) if recall_at_k else 0
    avg_precision = np.mean(precision_at_k) if precision_at_k else 0
    avg_ndcg = np.mean(ndcg_at_k) if ndcg_at_k else 0
    
    return {
        f'recall@{top_k}': avg_recall,
        f'precision@{top_k}': avg_precision,
        f'ndcg@{top_k}': avg_ndcg
    }

def main():
    """
    Main function to build the GNN graph and prepare data for citation prediction.
    """
    print("Building GNN model for citation recommendation...")
    
    # Add PyTorch Geometric classes to safe globals
    try:
        from torch.serialization import add_safe_globals
        from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
        add_safe_globals([Data, DataEdgeAttr, DataTensorAttr])
        print("Added PyTorch Geometric classes to safe globals")
    except ImportError:
        print("Could not import add_safe_globals or PyTorch Geometric classes")
    
    # Create cache directory
    os.makedirs('cache_gnn', exist_ok=True)
    
    # Define cache paths
    articles_cache_path = 'cache_gnn/articles.pkl'
    gnn_graph_cache_path = 'cache_gnn/gnn_citation_graph.gpickle'
    gnn_metadata_cache_path = 'cache_gnn/gnn_paper_metadata.json'
    author_graph_cache_path = 'cache_gnn/author_coauthor_graph.pkl'
    gnn_data_path = 'cache_gnn/gnn_data.pt'
    partial_citation_data_path = 'cache_gnn/partial_citation_prediction_data.pt'
    partial_model_path = 'cache_gnn/partial_citation_mlp_model.pt'
    
    # Step 1: Load articles (already handles caching)
    articles = load_articles()
    
    # Step 2: Build GNN graph (already handles caching)
    G, metadata, main_article_count = build_gnn_graph(articles)
    
    # Step 3: Build author coauthorship graph
    if os.path.exists(author_graph_cache_path):
        print("Loading author coauthorship graph from cache...")
        with open(author_graph_cache_path, 'rb') as f:
            author_coauthor_G = pkl.load(f)
    else:
        print("Building author coauthorship graph...")
        author_coauthor_G = process_articles_to_author_graphs(articles)
        # Cache the author graph
        print("Saving author coauthorship graph to cache...")
        with open(author_graph_cache_path, 'wb') as f:
            pkl.dump(author_coauthor_G, f)
    
    # Step 4: Convert to PyTorch Geometric format
    if os.path.exists(gnn_data_path):
        print("Loading PyTorch Geometric data from cache...")
        try:
            data = torch.load(gnn_data_path, weights_only=False)
            print("Successfully loaded PyTorch Geometric data")
        except Exception as e:
            print(f"Error loading PyTorch Geometric data: {e}")
            print("Converting graph to PyTorch Geometric format...")
            data = convert_to_pytorch_geometric(G, metadata)
            print("Saving PyTorch Geometric data to cache...")
            torch.save(data, gnn_data_path)
    else:
        print("Converting graph to PyTorch Geometric format...")
        data = convert_to_pytorch_geometric(G, metadata)
        print("Saving PyTorch Geometric data to cache...")
        torch.save(data, gnn_data_path)
    
    # Step 5: Prepare partial citation prediction data
    if os.path.exists(partial_citation_data_path):
        print("Loading partial citation prediction data from cache...")
        try:
            partial_citation_data = torch.load(partial_citation_data_path, weights_only=False)
            print("Successfully loaded partial citation prediction data")
        except Exception as e:
            print(f"Error loading partial citation prediction data: {e}")
            print("Preparing partial citation prediction data...")
            partial_citation_data = prepare_partial_citation_prediction_data(G, metadata, author_coauthor_G, observed_ratio=0.75)
            print("Saving partial citation prediction data to cache...")
            torch.save(partial_citation_data, partial_citation_data_path)
    else:
        print("Preparing partial citation prediction data...")
        partial_citation_data = prepare_partial_citation_prediction_data(G, metadata, author_coauthor_G, observed_ratio=0.75)
        print("Saving partial citation prediction data to cache...")
        torch.save(partial_citation_data, partial_citation_data_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 6: Train or load model for partial citation prediction
    model_exists = os.path.exists(partial_model_path)
    train_new_model = True
    
    if model_exists:
        user_input = input("A trained partial citation model already exists. Train a new model? (y/n): ")
        train_new_model = user_input.lower() == 'y'
    
    if train_new_model:
        # Train model
        print("Training partial citation prediction model...")
        model, history = train_partial_citation_mlp(
            data, 
            partial_citation_data, 
            device=device,
            batch_size=64,
            epochs=5,
            lr=0.001
        )
        
        # Plot training history
        print("Plotting training history...")
        plot_training_history(history)
        
        # Save model
        print("Saving model...")
        torch.save(model.state_dict(), partial_model_path)
    else:
        # Load existing model
        print("Loading trained model from cache...")
        node_embedding_dim = data.num_node_features
        edge_feature_dim = partial_citation_data['train']['edge_features'].shape[1]
        model = EnhancedCitationMLP(node_embedding_dim, edge_feature_dim)
        model.load_state_dict(torch.load(partial_model_path))
        model = model.to(device)
    
    # Step 7: Evaluate model
    print("Evaluating model...")
    metrics = evaluate_partial_citation_mlp(model, data, partial_citation_data, device)
    
    # Print metrics
    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    y_true = np.array(metrics['labels'])
    y_pred = np.array(metrics['probabilities']) > 0.5
    y_pred = y_pred.astype(int)
    plot_confusion_matrix(y_true, y_pred, normalize=False, title='Partial Citation Prediction Confusion Matrix')
    plot_confusion_matrix(y_true, y_pred, normalize=True, title='Partial Citation Prediction Normalized Confusion Matrix')
    
    # Step 8: Evaluate citation recommendations
    print("\nEvaluating citation recommendations...")
    recommendation_metrics = evaluate_citation_recommendations(model, data, partial_citation_data, G, metadata, top_k=10, device=device)
    
    print("\nCitation Recommendation Metrics:")
    for metric, value in recommendation_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Step 9: Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(model, data, partial_citation_data, device)
    
    print("\nEdge Feature Importance:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    print("GNN partial citation prediction pipeline complete!")

if __name__ == "__main__":
    main()
