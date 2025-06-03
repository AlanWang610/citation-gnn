#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GraphSAGE-based citation recommendation model.

This module implements a proper Graph Neural Network using GraphSAGE for learning
node representations from graph structure, while using SciBERT embedding similarity
as edge features for semantic understanding.
"""

import json
import os
import pickle as pkl
import random
import argparse
from collections import defaultdict
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

from create_gnn_graphs import (
    get_paper_id, standardize_journal_name, format_author_name,
    get_journal_encoding, should_filter_title, load_paper_embeddings
)

def load_scibert_model():
    """Load SciBERT model and tokenizer."""
    try:
        from transformers import AutoTokenizer, AutoModel
        print("Loading SciBERT model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        model.eval()
        print("SciBERT model loaded successfully")
        return tokenizer, model
    except ImportError:
        print("Transformers library not available, using dummy embeddings")
        return None, None

def load_pca_model(pca_path='embeddings/pca_model.pkl'):
    """Load PCA model for embedding reduction."""
    try:
        if os.path.exists(pca_path):
            print(f"Loading PCA model from {pca_path}...")
            with open(pca_path, 'rb') as f:
                pca_model = pkl.load(f)
            print(f"PCA model loaded successfully. Components: {pca_model.n_components_}")
            return pca_model
        else:
            print(f"PCA model not found at {pca_path}")
            return None
    except Exception as e:
        print(f"Error loading PCA model: {e}")
        return None

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    if not embedding1 or not embedding2:
        return 0.0
    
    # Convert to numpy arrays
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def create_year_scaler(G):
    """Create and fit a scaler for year normalization."""
    years = []
    for node_id in G.nodes():
        year = G.nodes[node_id].get('year', 0)
        if year > 0:  # Only include valid years
            years.append(year)
    
    if len(years) > 0:
        scaler = StandardScaler()
        years_array = np.array(years).reshape(-1, 1)
        scaler.fit(years_array)
        return scaler
    return None

def generate_node_features_for_new_paper(paper_json, G, metadata=None, normalize_year=True):
    """
    Generate node features for a new paper (inductive setting).
    Now focuses on basic features since SciBERT similarity is moved to edge features.
    """
    # Extract basic information
    title = paper_json.get('title', '')
    year_str = paper_json.get('published_date', '').split('-')[0]
    year = int(year_str) if year_str.isdigit() else 2020  # Default to 2020
    authors = paper_json.get('authors', [])
    journal = paper_json.get('journal', '')
    
    # Create paper ID
    paper_id = get_paper_id(title, authors, year_str)
    
    # Format authors
    formatted_authors = []
    for author in authors:
        formatted_name = format_author_name(author)
        if formatted_name:
            formatted_authors.append(formatted_name)
    
    # Get journal encoding
    journal_name = standardize_journal_name(journal)
    journal_encoding = get_journal_encoding(journal_name)
    
    # Normalize year if requested
    year_feature = year
    if normalize_year:
        year_scaler = create_year_scaler(G)
        if year_scaler:
            year_feature = year_scaler.transform([[year]])[0][0]
        else:
            # Fallback normalization
            years = [G.nodes[node].get('year', 2020) for node in G.nodes()]
            valid_years = [y for y in years if y > 0]
            if valid_years:
                mean_year = np.mean(valid_years)
                std_year = np.std(valid_years)
                if std_year > 0:
                    year_feature = (year - mean_year) / std_year
    
    # Count features (basic graph statistics)
    num_authors = len(formatted_authors)
    
    # Create feature vector: [year, journal_encoding (8 dims), num_authors, is_main_article]
    features = [year_feature] + journal_encoding + [num_authors, 1.0]  # 1.0 for is_main_article
    
    # Paper metadata
    paper_metadata = {
        'title': title,
        'journal': journal_name or '',
        'authors': "; ".join(formatted_authors),
        'is_main_article': True,
        'year': year
    }
    
    return paper_id, features, paper_metadata

class GraphSAGECitationModel(nn.Module):
    """
    GraphSAGE-based citation recommendation model.
    
    This model uses GraphSAGE to learn node representations from graph structure,
    then combines them with edge features (including SciBERT cosine similarity)
    for citation prediction.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, edge_feature_dim=6, 
                 aggregator='mean', dropout=0.2):
        super(GraphSAGECitationModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        
        # First layer
        self.sage_layers.append(SAGEConv(input_dim, hidden_dim, aggr=aggregator))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        # Edge feature processing
        self.edge_processor = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, src_nodes, tgt_nodes, edge_features):
        """
        Forward pass of the model.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            src_nodes: Source node indices [batch_size]
            tgt_nodes: Target node indices [batch_size]
            edge_features: Edge features [batch_size, edge_feature_dim]
        
        Returns:
            Citation predictions [batch_size, 1]
        """
        # Apply GraphSAGE layers
        h = x
        for i, layer in enumerate(self.sage_layers):
            h = layer(h, edge_index)
            if i < len(self.sage_layers) - 1:  # Don't apply activation after last layer
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Get embeddings for source and target nodes
        src_embeddings = h[src_nodes]  # [batch_size, hidden_dim]
        tgt_embeddings = h[tgt_nodes]  # [batch_size, hidden_dim]
        
        # Process edge features
        edge_processed = self.edge_processor(edge_features)  # [batch_size, hidden_dim // 2]
        
        # Combine all features
        combined = torch.cat([src_embeddings, tgt_embeddings, edge_processed], dim=1)
        
        # Make prediction
        return self.predictor(combined)

def extract_edge_features(source_id, target_id, G, metadata, paper_embeddings_dict, 
                         author_coauthor_G=None, observed_citations_only=False, 
                         observed_citations=None):
    """
    Extract edge features including SciBERT cosine similarity.
    
    Args:
        source_id: ID of the source paper
        target_id: ID of the target paper
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        paper_embeddings_dict: Dictionary mapping paper titles to SciBERT embeddings
        author_coauthor_G: Author co-authorship graph (optional)
        observed_citations_only: Whether to use only observed citations
        observed_citations: List of observed citation IDs
        
    Returns:
        Dictionary of edge features
    """
    features = {}
    
    # Get metadata for both papers
    source_metadata = metadata.get(source_id, {})
    target_metadata = metadata.get(target_id, {})
    
    # 1. SciBERT cosine similarity (NEW - main semantic feature)
    source_title = source_metadata.get('title', '')
    target_title = target_metadata.get('title', '')
    
    source_embedding = paper_embeddings_dict.get(source_title, [])
    target_embedding = paper_embeddings_dict.get(target_title, [])
    
    cosine_sim = calculate_cosine_similarity(source_embedding, target_embedding)
    features['scibert_cosine_similarity'] = cosine_sim
    
    # 2. Number of shared authors
    authors1 = set(source_metadata.get('authors', '').split('; '))
    authors2 = set(target_metadata.get('authors', '').split('; '))
    shared_authors = len(authors1.intersection(authors2))
    features['shared_authors'] = shared_authors
    
    # 3. Number of shared citations
    if observed_citations_only and observed_citations is not None:
        citations1 = set(observed_citations)
    else:
        citations1 = set(G.successors(source_id)) if source_id in G else set()
    
    citations2 = set(G.successors(target_id)) if target_id in G else set()
    shared_citations = len(citations1.intersection(citations2))
    features['shared_citations'] = shared_citations
    
    # 4. Journal similarity (same journal = 1, different = 0)
    source_journal = source_metadata.get('journal', '')
    target_journal = target_metadata.get('journal', '')
    journal_match = 1.0 if source_journal and target_journal and source_journal == target_journal else 0.0
    features['journal_match'] = journal_match
    
    # 5. Year difference (normalized)
    source_year = source_metadata.get('year', 2020)
    target_year = target_metadata.get('year', 2020)
    year_diff = abs(source_year - target_year) / 50.0  # Normalize by 50 years
    features['year_difference'] = min(year_diff, 1.0)  # Cap at 1.0
    
    # 6. Observed citation context (for partial citation prediction)
    if observed_citations_only and observed_citations is not None:
        target_citations = set(G.predecessors(target_id)) if target_id in G else set()
        observed_citing_target = len([c for c in observed_citations if c in target_citations])
        features['observed_citing_target'] = observed_citing_target / max(len(observed_citations), 1)
    else:
        features['observed_citing_target'] = 0.0
    
    return features

def load_articles(articles_file='articles.jsonl'):
    """Load articles from a JSONL file."""
    print(f"Loading articles from {articles_file}...")
    with open(articles_file, 'r') as f:
        articles = [json.loads(line) for line in f]
    print(f"Loaded {len(articles)} articles")
    return articles

def build_gnn_graph(articles, embedding_file='embeddings/papers_with_embeddings.pkl'):
    """
    Build a graph for GraphSAGE-based citation recommendation.
    Node features now exclude SciBERT embeddings (moved to edge features).
    """
    print("Building GraphSAGE citation graph...")
    
    # Load paper embeddings for edge features
    paper_embeddings_dict = load_paper_embeddings(embedding_file)
    
    # Create directed graph
    G = nx.DiGraph()
    metadata = {}
    main_article_count = 0
    
    # First pass: Add all nodes with basic features
    print("Adding nodes with basic features...")
    for article in tqdm(articles):
        if not article.get('title') or not article.get('published_date') or not article.get('authors'):
            continue
            
        if should_filter_title(article.get('title', '')):
            continue
            
        title = article['title']
        year = article['published_date'].split('-')[0]
        authors = article.get('authors', [])
        
        article_id = get_paper_id(title, authors, year)
        if not article_id:
            continue
            
        # Format authors
        formatted_authors = []
        for author in authors:
            formatted_name = format_author_name(author)
            if formatted_name:
                formatted_authors.append(formatted_name)
        
        # Get journal encoding
        journal_name = standardize_journal_name(article.get('journal', ''))
        journal_encoding = get_journal_encoding(journal_name)
        
        # Create basic node features: [year, journal_encoding (8), num_authors, is_main_article]
        year_int = int(year) if year.isdigit() else 2020
        num_authors = len(formatted_authors)
        node_features = [year_int] + journal_encoding + [num_authors, 1.0]  # is_main_article = 1
        
        G.add_node(article_id, features=node_features, year=year_int)
        
        metadata[article_id] = {
            'title': title,
            'journal': journal_name or '',
            'authors': "; ".join(formatted_authors),
            'is_main_article': True,
            'year': year_int
        }
        
        main_article_count += 1
        
        # Process references
        if 'references' in article:
            for ref in article['references']:
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
                
                if ref_id:
                    # Add reference node if not exists
                    if ref_id not in G:
                        ref_formatted_authors = []
                        ref_authors = ref.get('authors', [])
                        for author in ref_authors:
                            formatted_name = format_author_name(author)
                            if formatted_name:
                                ref_formatted_authors.append(formatted_name)
                        
                        ref_journal_name = standardize_journal_name(ref.get('journal', ''))
                        ref_journal_encoding = get_journal_encoding(ref_journal_name)
                        
                        ref_year = ref.get('year', '')
                        ref_year_int = int(ref_year) if str(ref_year).isdigit() else 2020
                        ref_num_authors = len(ref_formatted_authors)
                        
                        ref_node_features = [ref_year_int] + ref_journal_encoding + [ref_num_authors, 0.0]  # is_main_article = 0
                        
                        G.add_node(ref_id, features=ref_node_features, year=ref_year_int)
                        
                        metadata[ref_id] = {
                            'title': ref_title,
                            'journal': ref_journal_name or '',
                            'authors': "; ".join(ref_formatted_authors),
                            'is_main_article': False,
                            'year': ref_year_int
                        }
                    
                    # Add citation edge
                    G.add_edge(article_id, ref_id)
    
    # Normalize year features
    print("Normalizing year features...")
    year_scaler = create_year_scaler(G)
    if year_scaler:
        for node_id in G.nodes():
            features = G.nodes[node_id]['features']
            year = features[0]
            normalized_year = year_scaler.transform([[year]])[0][0]
            features[0] = normalized_year
            G.nodes[node_id]['features'] = features
    
    print(f"Created GraphSAGE citation graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Processed {main_article_count} main articles")
    
    return G, metadata, main_article_count, paper_embeddings_dict

def convert_to_pytorch_geometric(G, metadata):
    """Convert NetworkX graph to PyTorch Geometric format for GraphSAGE."""
    print("Converting to PyTorch Geometric format...")
    
    # Create node mapping
    node_ids = list(G.nodes())
    node_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # Extract node features
    node_features = []
    for node_id in node_ids:
        features = G.nodes[node_id]['features']
        node_features.append(features)
    
    # Convert to tensor
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge index
    edges = list(G.edges())
    edge_index = torch.tensor([[node_mapping[src], node_mapping[dst]] for src, dst in edges], 
                             dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=node_features_tensor, edge_index=edge_index)
    data.node_mapping = node_mapping
    data.reverse_mapping = {i: node_id for node_id, i in node_mapping.items()}
    
    print(f"Created PyTorch Geometric Data with {data.num_nodes} nodes and {data.num_edges} edges")
    print(f"Node feature dimensions: {data.num_node_features}")
    
    return data

def prepare_citation_prediction_data(G, metadata, paper_embeddings_dict, author_coauthor_G=None, 
                                   split_ratio=0.8, random_seed=42):
    """Prepare data for citation prediction with GraphSAGE."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    main_articles = [node for node in G.nodes() if metadata[node].get('is_main_article', True)]
    print(f"Found {len(main_articles)} main articles")
    
    positive_examples = []
    negative_examples = []
    
    for main_article in tqdm(main_articles, desc="Creating citation examples"):
        cited_papers = list(G.successors(main_article))
        
        # Positive examples
        for cited_paper in cited_papers:
            edge_features = extract_edge_features(
                main_article, cited_paper, G, metadata, paper_embeddings_dict, author_coauthor_G
            )
            positive_examples.append((main_article, cited_paper, 1, edge_features))
        
        # Negative examples
        all_papers = list(G.nodes())
        non_cited_papers = [p for p in all_papers if p != main_article and p not in cited_papers]
        
        num_negative_samples = min(len(non_cited_papers), 3 * len(cited_papers))
        if len(non_cited_papers) > num_negative_samples:
            non_cited_papers = random.sample(non_cited_papers, num_negative_samples)
        
        for non_cited_paper in non_cited_papers:
            edge_features = extract_edge_features(
                main_article, non_cited_paper, G, metadata, paper_embeddings_dict, author_coauthor_G
            )
            negative_examples.append((main_article, non_cited_paper, 0, edge_features))
    
    print(f"Created {len(positive_examples)} positive and {len(negative_examples)} negative examples")
    
    # Combine and shuffle
    all_examples = positive_examples + negative_examples
    random.shuffle(all_examples)
    
    # Split data
    split_idx = int(len(all_examples) * split_ratio)
    train_examples = all_examples[:split_idx]
    test_examples = all_examples[split_idx:]
    
    # Create node mapping
    node_ids = list(G.nodes())
    node_mapping = {node_id: i for i, node_id in enumerate(node_ids)}
    
    # Convert to tensors
    def examples_to_tensors(examples):
        source_nodes = [node_mapping[src] for src, _, _, _ in examples]
        target_nodes = [node_mapping[dst] for _, dst, _, _ in examples]
        labels = [label for _, _, label, _ in examples]
        edge_features = [list(features.values()) for _, _, _, features in examples]
        
        return {
            'source_nodes': torch.tensor(source_nodes, dtype=torch.long),
            'target_nodes': torch.tensor(target_nodes, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float),
            'edge_features': torch.tensor(edge_features, dtype=torch.float),
            'examples': examples
        }
    
    train_data = examples_to_tensors(train_examples)
    test_data = examples_to_tensors(test_examples)
    
    # Get edge feature names
    edge_feature_names = list(train_examples[0][3].keys()) if train_examples else []
    
    return {
        'train': train_data,
        'test': test_data,
        'node_mapping': node_mapping,
        'reverse_mapping': {i: node_id for node_id, i in node_mapping.items()},
        'edge_feature_names': edge_feature_names
    }

def clear_cache(cache_dir='cache_gnn'):
    """Clear all cached files."""
    import shutil
    if os.path.exists(cache_dir):
        print(f"Clearing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("Cache cleared successfully")
    else:
        print("Cache directory does not exist")

def get_cache_status(cache_dir='cache_gnn'):
    """Check which cache files exist and their sizes."""
    cache_files = [
        'articles.pkl',
        'gnn_citation_graph.gpickle',
        'gnn_paper_metadata.json',
        'paper_embeddings_dict.pkl',
        'gnn_data.pt',
        'citation_prediction_data.pt',
        'graphsage_citation_model.pt',
        'training_history.pkl'
    ]
    
    print("Cache status:")
    if not os.path.exists(cache_dir):
        print(f"  Cache directory '{cache_dir}' does not exist")
        return
    
    for filename in cache_files:
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename}")

def optimize_batch_size_for_gpu(data, citation_data, device='cuda', start_batch_size=64):
    """Automatically find the optimal batch size for the available GPU memory."""
    if device == 'cpu':
        return min(start_batch_size * 2, 512)  # Larger batches for CPU
    
    if not torch.cuda.is_available():
        return start_batch_size
    
    print("Finding optimal batch size for GPU...")
    
    # Get GPU memory info
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"Total GPU memory: {total_memory / 1024**3:.1f} GB")
    
    # Test different batch sizes
    input_dim = data.num_node_features
    edge_feature_dim = citation_data['train']['edge_features'].shape[1]
    
    model = GraphSAGECitationModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        edge_feature_dim=edge_feature_dim,
        dropout=0.2
    ).to(device)
    
    data = data.to(device)
    criterion = nn.BCELoss()
    
    batch_size = start_batch_size
    max_batch_size = start_batch_size
    
    while batch_size <= 2048:  # Don't go too high
        try:
            torch.cuda.empty_cache()
            
            # Create dummy batch
            train_data = citation_data['train']
            if len(train_data['source_nodes']) < batch_size:
                break
                
            batch_src = train_data['source_nodes'][:batch_size].to(device)
            batch_tgt = train_data['target_nodes'][:batch_size].to(device)
            batch_edge = train_data['edge_features'][:batch_size].to(device)
            batch_labels = train_data['labels'][:batch_size].to(device)
            
            # Forward pass
            outputs = model(data.x, data.edge_index, batch_src, batch_tgt, batch_edge)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Backward pass
            loss.backward()
            
            max_batch_size = batch_size
            print(f"Batch size {batch_size}: OK")
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: Out of memory")
                break
            else:
                raise e
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    print(f"Optimal batch size: {max_batch_size}")
    return max_batch_size

def train_graphsage_model(data, citation_data, device='cuda', batch_size=64, epochs=50, lr=0.001, 
                         save_checkpoints=True, checkpoint_every=5, optimize_batch_size=True,
                         use_mixed_precision=True, gradient_accumulation_steps=1, 
                         num_workers=12, pin_memory=True):
    """
    Train the GraphSAGE citation model with comprehensive optimizations.
    
    Args:
        optimize_batch_size: Automatically find optimal batch size for GPU
        use_mixed_precision: Use automatic mixed precision training (faster on modern GPUs)
        gradient_accumulation_steps: Accumulate gradients over multiple batches
        num_workers: Number of workers for data loading
        pin_memory: Use pinned memory for faster GPU transfer
    """
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        use_mixed_precision = False  # Mixed precision only works on GPU
    
    # Optimize batch size if requested
    if optimize_batch_size and device == 'cuda':
        batch_size = optimize_batch_size_for_gpu(data, citation_data, device, batch_size)
    
    # Adjust num_workers for CPU vs GPU
    if device == 'cpu':
        import multiprocessing
        num_workers = min(num_workers, multiprocessing.cpu_count())
    else:
        num_workers = min(num_workers, 12)  # Use up to 12 workers for GPU systems
    
    print(f"Training configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Mixed precision: {use_mixed_precision}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Data loading workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    
    # Move data to device
    data = data.to(device)
    
    # Get dimensions
    input_dim = data.num_node_features
    edge_feature_dim = citation_data['train']['edge_features'].shape[1]
    
    # Initialize model with optimizations
    model = GraphSAGECitationModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        edge_feature_dim=edge_feature_dim,
        dropout=0.2
    ).to(device)
    
    # Enable compilation for PyTorch 2.0+ (significant speedup)
    if hasattr(torch, 'compile') and device == 'cuda':
        print("Compiling model with torch.compile for additional speedup...")
        model = torch.compile(model)
    
    # Loss and optimizer with improved settings
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, eps=1e-8)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Training data with optimized DataLoader
    train_data = citation_data['train']
    train_dataset = TensorDataset(
        train_data['source_nodes'],
        train_data['target_nodes'],
        train_data['edge_features'],
        train_data['labels']
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and device == 'cuda',
        drop_last=True,  # Avoid issues with batch norm on small final batches
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    # Training history with more metrics
    history = {
        'loss': [], 'val_loss': [], 'val_auc': [], 
        'learning_rate': [], 'epoch_time': [], 'batch_time': []
    }
    
    # Create checkpoint directory
    if save_checkpoints:
        os.makedirs('cache_gnn/checkpoints', exist_ok=True)
    
    print(f"Training GraphSAGE model for {epochs} epochs...")
    print(f"Training batches per epoch: {len(train_loader)}")
    
    # Training loop with optimizations
    best_val_auc = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        batch_times = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (batch_src, batch_tgt, batch_edge, batch_labels) in enumerate(progress_bar):
            batch_start_time = time.time()
            
            batch_src = batch_src.to(device, non_blocking=True)
            batch_tgt = batch_tgt.to(device, non_blocking=True)
            batch_edge = batch_edge.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(data.x, data.edge_index, batch_src, batch_tgt, batch_edge)
                    loss = criterion(outputs.squeeze(), batch_labels)
                    loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation
            else:
                outputs = model(data.x, data.edge_index, batch_src, batch_tgt, batch_edge)
                loss = criterion(outputs.squeeze(), batch_labels)
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with gradient accumulation
            if use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Track timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            avg_batch_time = np.mean(batch_times[-10:])  # Moving average of last 10 batches
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'batch_time': f'{avg_batch_time:.3f}s',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Epoch metrics
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times)
        
        history['loss'].append(avg_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['epoch_time'].append(epoch_time)
        history['batch_time'].append(avg_batch_time)
        
        # Evaluate with optimized batch size
        eval_batch_size = min(batch_size * 2, 1024)  # Larger batch for evaluation
        val_metrics = evaluate_graphsage_model(model, data, citation_data, device, eval_batch_size)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc'].append(val_metrics['auc'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            if save_checkpoints:
                best_model_path = 'cache_gnn/best_graphsage_model.pt'
                torch.save(model.state_dict(), best_model_path)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
        print(f"  Epoch time: {epoch_time:.1f}s | Avg batch time: {avg_batch_time:.3f}s")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        if save_checkpoints and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = f'cache_gnn/checkpoints/model_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'loss': avg_loss,
                'val_auc': val_metrics['auc'],
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    return model, history

def evaluate_graphsage_model(model, data, citation_data, device='cuda', batch_size=64, 
                           num_workers=12, pin_memory=True):
    """Evaluate the GraphSAGE model with optimized data loading."""
    model.eval()
    
    # Adjust num_workers for evaluation
    if device == 'cpu':
        import multiprocessing
        num_workers = min(num_workers, multiprocessing.cpu_count())
    else:
        num_workers = min(num_workers, 8)  # Use up to 8 workers for evaluation (lighter load)
    
    # Test data with optimized DataLoader
    test_data = citation_data['test']
    test_dataset = TensorDataset(
        test_data['source_nodes'],
        test_data['target_nodes'],
        test_data['edge_features'],
        test_data['labels']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and device == 'cuda',
        persistent_workers=num_workers > 0
    )
    
    criterion = nn.BCELoss()
    
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_src, batch_tgt, batch_edge, batch_labels in test_loader:
            batch_src = batch_src.to(device, non_blocking=True)
            batch_tgt = batch_tgt.to(device, non_blocking=True)
            batch_edge = batch_edge.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            outputs = model(data.x, data.edge_index, batch_src, batch_tgt, batch_edge)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            total_loss += loss.item()
            
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = batch_labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def inductive_recommend_citations(paper_json, G, metadata, model, data, paper_embeddings_dict,
                                author_coauthor_G=None, top_k=10, exclude_observed=True, 
                                device='cpu', include_feature_importance=False):
    """Recommend citations for a new paper using GraphSAGE (inductive setting)."""
    # Generate features for new paper
    paper_id, paper_features, paper_metadata = generate_node_features_for_new_paper(paper_json, G, metadata)
    
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
    
    # Get candidates
    candidates = [node for node in G.nodes()]
    if exclude_observed:
        candidates = [c for c in candidates if c not in observed_citations]
    
    # Create temporary extended graph with new paper
    extended_data = data.clone()
    
    # Add new paper features to node features
    new_node_features = torch.cat([
        extended_data.x,
        torch.tensor([paper_features], dtype=torch.float).to(device)
    ], dim=0)
    
    extended_data.x = new_node_features
    new_node_idx = len(data.node_mapping)
    
    # Move model and data to device
    model = model.to(device)
    extended_data = extended_data.to(device)
    
    # Batch prediction
    batch_size = 1000
    all_probs = []
    all_edge_features = []
    
    for i in range(0, len(candidates), batch_size):
        batch_candidates = candidates[i:i+batch_size]
        batch_tgt_idxs = [data.node_mapping[c] for c in batch_candidates]
        
        # Extract edge features
        edge_features = []
        for candidate in batch_candidates:
            features = extract_edge_features(
                paper_id, candidate, G, metadata, paper_embeddings_dict,
                author_coauthor_G, observed_citations_only=True, 
                observed_citations=observed_citations
            )
            edge_features.append(list(features.values()))
        
        all_edge_features.extend(edge_features)
        
        # Create tensors
        src_tensor = torch.tensor([new_node_idx] * len(batch_candidates), dtype=torch.long).to(device)
        tgt_tensor = torch.tensor(batch_tgt_idxs, dtype=torch.long).to(device)
        edge_tensor = torch.tensor(edge_features, dtype=torch.float).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(extended_data.x, extended_data.edge_index, src_tensor, tgt_tensor, edge_tensor)
            probs = outputs.squeeze().cpu().numpy()
        
        all_probs.extend(probs)
    
    # Rank and get top-k
    ranked_indices = np.argsort(-np.array(all_probs))
    
    recommendations = []
    for i, idx in enumerate(ranked_indices[:top_k]):
        candidate = candidates[idx]
        candidate_metadata = metadata.get(candidate, {})
        
        recommendation = {
            'rank': i + 1,
            'id': candidate,
            'title': candidate_metadata.get('title', ''),
            'authors': candidate_metadata.get('authors', ''),
            'journal': candidate_metadata.get('journal', ''),
            'score': float(all_probs[idx])
        }
        
        recommendations.append(recommendation)
    
    return recommendations, paper_id, observed_citations, paper_metadata

def inductive_evaluate_with_held_out(paper_json, G, metadata, model, data, paper_embeddings_dict,
                                   author_coauthor_G=None, observed_ratio=0.75, top_k=10, 
                                   device='cpu', random_seed=111, include_feature_importance=False):
    """Evaluate citation recommendations for a new paper with held-out citations."""
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
    
    # Set random seed for reproducible results
    random.seed(random_seed)
    
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
        data,
        paper_embeddings_dict,
        author_coauthor_G,
        top_k=top_k,
        exclude_observed=True,
        device=device,
        include_feature_importance=include_feature_importance
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
            dcg += 1 / np.log2(i + 2)
    
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

def main(optimize_batch_size=True, use_mixed_precision=True, gradient_accumulation_steps=1, 
         num_workers=12, pin_memory=True, epochs=1, batch_size=64, lr=0.001, device='auto'):
    """Main function to build and train the GraphSAGE citation model with optimizations."""
    print("Building GraphSAGE citation recommendation model...")
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create cache directory
    os.makedirs('cache_gnn', exist_ok=True)
    
    # Step 1: Load articles (with caching)
    articles_cache_path = 'cache_gnn/articles.pkl'
    if os.path.exists(articles_cache_path):
        print("Loading articles from cache...")
        with open(articles_cache_path, 'rb') as f:
            articles = pkl.load(f)
        print(f"Loaded {len(articles)} articles from cache")
    else:
        print("Loading articles from file...")
        articles = load_articles()
        # Save articles cache immediately
        print("Saving articles to cache...")
        with open(articles_cache_path, 'wb') as f:
            pkl.dump(articles, f)
        print("Articles cached successfully")
    
    # Step 2: Build graph with embeddings (with caching)
    graph_cache_path = 'cache_gnn/gnn_citation_graph.gpickle'
    metadata_cache_path = 'cache_gnn/gnn_paper_metadata.json'
    embeddings_cache_path = 'cache_gnn/paper_embeddings_dict.pkl'
    
    if (os.path.exists(graph_cache_path) and 
        os.path.exists(metadata_cache_path) and 
        os.path.exists(embeddings_cache_path)):
        print("Loading graph, metadata, and embeddings from cache...")
        with open(graph_cache_path, 'rb') as f:
            G = pkl.load(f)
        with open(metadata_cache_path, 'r') as f:
            metadata = json.load(f)
        with open(embeddings_cache_path, 'rb') as f:
            paper_embeddings_dict = pkl.load(f)
        main_article_count = len([node for node in metadata if metadata[node].get('is_main_article', False)])
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges from cache")
    else:
        print("Building graph with embeddings...")
        G, metadata, main_article_count, paper_embeddings_dict = build_gnn_graph(articles)
        # Save graph components immediately
        print("Saving graph components to cache...")
        with open(graph_cache_path, 'wb') as f:
            pkl.dump(G, f)
        with open(metadata_cache_path, 'w') as f:
            json.dump(metadata, f)
        with open(embeddings_cache_path, 'wb') as f:
            pkl.dump(paper_embeddings_dict, f)
        print("Graph components cached successfully")
    
    # Step 3: Convert to PyTorch Geometric (with caching)
    data_cache_path = 'cache_gnn/gnn_data.pt'
    if os.path.exists(data_cache_path):
        print("Loading PyTorch Geometric data from cache...")
        data = torch.load(data_cache_path, weights_only=False)
        print(f"Loaded PyTorch Geometric data with {data.num_nodes} nodes from cache")
    else:
        print("Converting to PyTorch Geometric format...")
        data = convert_to_pytorch_geometric(G, metadata)
        # Save data immediately
        print("Saving PyTorch Geometric data to cache...")
        torch.save(data, data_cache_path)
        print("PyTorch Geometric data cached successfully")
    
    # Step 4: Prepare citation prediction data (with caching)
    citation_data_cache_path = 'cache_gnn/citation_prediction_data.pt'
    if os.path.exists(citation_data_cache_path):
        print("Loading citation prediction data from cache...")
        citation_data = torch.load(citation_data_cache_path, weights_only=False)
        print(f"Loaded citation data with {len(citation_data['train']['examples'])} train and {len(citation_data['test']['examples'])} test examples from cache")
    else:
        print("Preparing citation prediction data...")
        citation_data = prepare_citation_prediction_data(G, metadata, paper_embeddings_dict)
        # Save citation data immediately
        print("Saving citation prediction data to cache...")
        torch.save(citation_data, citation_data_cache_path)
        print("Citation prediction data cached successfully")
    
    # Step 5: Train model (with caching and checkpoints)
    model_cache_path = 'cache_gnn/graphsage_citation_model.pt'
    training_history_cache_path = 'cache_gnn/training_history.pkl'
    
    if os.path.exists(model_cache_path):
        print("Loading trained model from cache...")
        # Get dimensions to initialize model
        input_dim = data.num_node_features
        edge_feature_dim = citation_data['train']['edge_features'].shape[1]
        
        # Initialize model
        model = GraphSAGECitationModel(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=2,
            edge_feature_dim=edge_feature_dim,
            dropout=0.2
        )
        model.load_state_dict(torch.load(model_cache_path))
        
        # Load training history if available
        if os.path.exists(training_history_cache_path):
            with open(training_history_cache_path, 'rb') as f:
                history = pkl.load(f)
        else:
            history = {'loss': [], 'val_loss': [], 'val_auc': []}
        
        print("Trained model loaded from cache")
    else:
        print("Training GraphSAGE model with optimizations...")
        print(f"Using device: {device}")
        
        model, history = train_graphsage_model(
            data, 
            citation_data, 
            device=device, 
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimize_batch_size=optimize_batch_size,
            use_mixed_precision=use_mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Save model and history immediately after training
        print("Saving trained model to cache...")
        torch.save(model.state_dict(), model_cache_path)
        with open(training_history_cache_path, 'wb') as f:
            pkl.dump(history, f)
        print("Trained model cached successfully")
    
    print("All steps completed successfully!")
    print("Cache summary:")
    print(f"  - Articles: {articles_cache_path}")
    print(f"  - Graph: {graph_cache_path}")
    print(f"  - Metadata: {metadata_cache_path}")
    print(f"  - Embeddings: {embeddings_cache_path}")
    print(f"  - PyTorch Geometric data: {data_cache_path}")
    print(f"  - Citation prediction data: {citation_data_cache_path}")
    print(f"  - Trained model: {model_cache_path}")
    print(f"  - Training history: {training_history_cache_path}")
    
    return model, data, citation_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GraphSAGE Citation Recommendation Model')
    parser.add_argument('--action', choices=['train', 'clear_cache', 'check_cache'], 
                       default='train', help='Action to perform')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Initial batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto', 
                       help='Device to use for training')
    parser.add_argument('--save_checkpoints', action='store_true', default=True,
                       help='Save model checkpoints during training')
    parser.add_argument('--checkpoint_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # New optimization parameters
    parser.add_argument('--optimize_batch_size', action='store_true', default=True,
                       help='Automatically find optimal batch size for GPU')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use automatic mixed precision training (faster on modern GPUs)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Accumulate gradients over multiple batches')
    parser.add_argument('--num_workers', type=int, default=12,
                       help='Number of workers for data loading')
    parser.add_argument('--no_pin_memory', action='store_true',
                       help='Disable pinned memory for GPU transfer')
    
    args = parser.parse_args()
    
    if args.action == 'clear_cache':
        clear_cache()
    elif args.action == 'check_cache':
        get_cache_status()
    else:  # train
        # Determine device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
            
        pin_memory = not args.no_pin_memory
        
        print(f"Starting optimized training with device: {device}")
        print(f"Epochs: {args.epochs}, Initial batch size: {args.batch_size}, Learning rate: {args.lr}")
        print(f"Optimizations:")
        print(f"  Auto batch size: {args.optimize_batch_size}")
        print(f"  Mixed precision: {args.mixed_precision}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Data loading workers: {args.num_workers}")
        print(f"  Pin memory: {pin_memory}")
        
        # Check cache status before starting
        print("\n" + "="*50)
        print("CACHE STATUS BEFORE TRAINING")
        print("="*50)
        get_cache_status()
        print("="*50 + "\n")
        
        # Run main training pipeline with optimizations
        model, data, citation_data = main(
            optimize_batch_size=args.optimize_batch_size,
            use_mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        get_cache_status()
        print("="*50) 
