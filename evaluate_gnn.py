#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate GNN model for citation recommendation.

This script loads a trained GNN model and recommends additional citations
for a paper with partial citation information.
"""

import json
import os
import pickle as pkl
import random
import argparse
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from create_gnn_graphs import (
    get_paper_id, standardize_journal_name, format_author_name,
    get_journal_encoding, should_filter_title
)
from gnn_model import (
    EnhancedCitationMLP, extract_edge_features
)

def load_model_and_data(cache_dir='cache_gnn'):
    """
    Load the trained model and graph data from cache.
    
    Args:
        cache_dir: Directory containing cached files
        
    Returns:
        Tuple of (model, graph, metadata, node_features, node_mapping)
    """
    print("Loading model and data from cache...")
    
    # Define cache paths
    gnn_graph_path = os.path.join(cache_dir, 'gnn_citation_graph.gpickle')
    gnn_metadata_path = os.path.join(cache_dir, 'gnn_paper_metadata.json')
    gnn_data_path = os.path.join(cache_dir, 'gnn_data.pt')
    model_path = os.path.join(cache_dir, 'partial_citation_mlp_model.pt')
    author_graph_path = os.path.join(cache_dir, 'author_coauthor_graph.pkl')
    
    # Check if all required files exist
    required_files = [gnn_graph_path, gnn_metadata_path, gnn_data_path, model_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Load graph and metadata
    print("Loading citation graph...")
    with open(gnn_graph_path, 'rb') as f:
        G = pkl.load(f)
    
    print("Loading paper metadata...")
    with open(gnn_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load author coauthorship graph if available
    author_coauthor_G = None
    if os.path.exists(author_graph_path):
        print("Loading author coauthorship graph...")
        with open(author_graph_path, 'rb') as f:
            author_coauthor_G = pkl.load(f)
    
    # Load PyTorch Geometric data
    print("Loading PyTorch Geometric data...")
    data = torch.load(gnn_data_path, weights_only=False)
    
    # Get node features and mapping
    node_features = data.x
    node_mapping = data.node_mapping if hasattr(data, 'node_mapping') else None
    reverse_mapping = data.reverse_mapping if hasattr(data, 'reverse_mapping') else None
    
    # Load model
    print("Loading trained model...")
    node_embedding_dim = data.num_node_features
    # Determine edge feature dimension from a sample edge
    sample_edge_features = extract_edge_features(
        list(G.nodes())[0], 
        list(G.successors(list(G.nodes())[0]))[0] if list(G.successors(list(G.nodes())[0])) else list(G.nodes())[1],
        G, 
        metadata,
        author_coauthor_G
    )
    edge_feature_dim = len(sample_edge_features)
    
    # Initialize and load model
    model = EnhancedCitationMLP(node_embedding_dim, edge_feature_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Successfully loaded model and data")
    
    return model, G, metadata, node_features, node_mapping, reverse_mapping, author_coauthor_G

def parse_paper(paper_json, G, metadata):
    """
    Parse a paper JSON and extract relevant information.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        
    Returns:
        Tuple of (paper_id, observed_citations, paper_data)
    """
    # Extract paper information
    title = paper_json.get('title')
    year = paper_json.get('published_date', '').split('-')[0]
    authors = paper_json.get('authors', [])
    
    # Create paper ID
    paper_id = get_paper_id(title, authors, year)
    
    # Check if paper exists in graph
    if paper_id not in G:
        print(f"Warning: Paper '{title}' not found in graph")
        return None, [], {}
    
    # Extract references
    references = paper_json.get('references', [])
    observed_citations = []
    
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
            observed_citations.append(ref_id)
    
    # Create paper data dictionary
    paper_data = {
        'id': paper_id,
        'title': title,
        'year': year,
        'authors': authors,
        'observed_citations': observed_citations
    }
    
    return paper_id, observed_citations, paper_data

def recommend_citations(paper_id, observed_citations, G, metadata, model, node_features, 
                       node_mapping, reverse_mapping, author_coauthor_G, top_k=10, 
                       exclude_observed=True, device='cpu'):
    """
    Recommend citations for a paper based on partial citation information.
    
    Args:
        paper_id: ID of the paper
        observed_citations: List of observed citation IDs
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        model: Trained EnhancedCitationMLP model
        node_features: Node feature tensor
        node_mapping: Mapping from node IDs to indices
        reverse_mapping: Mapping from indices to node IDs
        author_coauthor_G: Author coauthorship graph
        top_k: Number of top recommendations to return
        exclude_observed: Whether to exclude observed citations from recommendations
        device: Device to run the model on
        
    Returns:
        List of dictionaries containing recommended citations
    """
    # Check if paper exists in graph
    if paper_id not in G:
        print(f"Error: Paper ID {paper_id} not found in graph")
        return []
    
    # Move model and node features to device
    model = model.to(device)
    node_features = node_features.to(device)
    
    # Get source node index
    src_idx = node_mapping[paper_id]
    
    # Get candidate papers (all papers except the source paper)
    candidates = [node for node in G.nodes() if node != paper_id]
    
    # Exclude observed citations if requested
    if exclude_observed:
        candidates = [c for c in candidates if c not in observed_citations]
    
    # Prepare batches for prediction
    batch_size = 1000
    all_probs = []
    
    # Process candidates in batches
    for i in range(0, len(candidates), batch_size):
        batch_candidates = candidates[i:i+batch_size]
        batch_idxs = [node_mapping[c] for c in batch_candidates]
        
        # Get source embeddings (repeated for each candidate)
        src_embeddings = node_features[src_idx].repeat(len(batch_candidates), 1)
        
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
    
    return recommendations

def evaluate_with_held_out(paper_json, G, metadata, model, node_features, node_mapping, 
                          reverse_mapping, author_coauthor_G, observed_ratio=0.75, 
                          top_k=10, device='cpu'):
    """
    Evaluate citation recommendations against held-out citations.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        model: Trained EnhancedCitationMLP model
        node_features: Node feature tensor
        node_mapping: Mapping from node IDs to indices
        reverse_mapping: Mapping from indices to node IDs
        author_coauthor_G: Author coauthorship graph
        observed_ratio: Ratio of citations to observe
        top_k: Number of top recommendations to consider
        device: Device to run the model on
        
    Returns:
        Dictionary of evaluation metrics and recommendations
    """
    # Parse paper
    paper_id, all_citations, paper_data = parse_paper(paper_json, G, metadata)
    
    if paper_id is None:
        return {'error': 'Paper not found in graph'}
    
    if len(all_citations) < 2:
        return {'error': 'Paper has too few citations for evaluation'}
    
    # Split citations into observed and held-out sets
    random.shuffle(all_citations)
    num_observed = max(1, int(len(all_citations) * observed_ratio))
    observed_citations = all_citations[:num_observed]
    held_out_citations = all_citations[num_observed:]
    
    print(f"Paper: {paper_data['title']}")
    print(f"Total citations: {len(all_citations)}")
    print(f"Observed citations: {len(observed_citations)}")
    print(f"Held-out citations: {len(held_out_citations)}")
    
    # Get recommendations
    recommendations = recommend_citations(
        paper_id, 
        observed_citations, 
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
        'paper_title': paper_data['title'],
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

def recommend_for_paper(paper_json, G, metadata, model, node_features, node_mapping, 
                       reverse_mapping, author_coauthor_G, top_k=10, device='cpu'):
    """
    Recommend citations for a paper based on its existing citations.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        model: Trained EnhancedCitationMLP model
        node_features: Node feature tensor
        node_mapping: Mapping from node IDs to indices
        reverse_mapping: Mapping from indices to node IDs
        author_coauthor_G: Author coauthorship graph
        top_k: Number of top recommendations to return
        device: Device to run the model on
        
    Returns:
        Dictionary containing recommendations and paper information
    """
    # Parse paper
    paper_id, observed_citations, paper_data = parse_paper(paper_json, G, metadata)
    
    if paper_id is None:
        return {'error': 'Paper not found in graph'}
    
    print(f"Paper: {paper_data['title']}")
    print(f"Existing citations: {len(observed_citations)}")
    
    # Get recommendations
    recommendations = recommend_citations(
        paper_id, 
        observed_citations, 
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
    
    # Return recommendations and paper information
    return {
        'paper_id': paper_id,
        'paper_title': paper_data['title'],
        'existing_citations': len(observed_citations),
        'recommendations': recommendations
    }

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate GNN model for citation recommendation')
    parser.add_argument('--input', type=str, help='Path to input JSON file containing paper information')
    parser.add_argument('--output', type=str, default='recommendations.json', help='Path to output JSON file')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top recommendations to return')
    parser.add_argument('--mode', type=str, default='recommend', choices=['recommend', 'evaluate'], 
                        help='Mode: recommend (use all citations) or evaluate (hold out some citations)')
    parser.add_argument('--observed-ratio', type=float, default=0.75, 
                        help='Ratio of citations to observe in evaluation mode')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Device to run the model on')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Load model and data
    model, G, metadata, node_features, node_mapping, reverse_mapping, author_coauthor_G = load_model_and_data()
    
    # Load input paper
    if args.input:
        with open(args.input, 'r') as f:
            paper_json = json.load(f)
    else:
        # Use example paper
        paper_json = {
            "doi": "10.1016/j.jfineco.2024.103897", 
            "type": "journal-article", 
            "published_date": "2024-09-01", 
            "title": "Are cryptos different? Evidence from retail trading", 
            "journal": "Journal of Financial Economics", 
            "abstract": None, 
            "volume": "159", 
            "issue": None, 
            "authors": [["Shimon", "Kogan", None], ["Igor", "Makarov", None], ["Marina", "Niessner", None], ["Antoinette", "Schoar", None]], 
            "references": [{"reference_type": "book", "doi": "10.1016/b978-0-12-822927-9.00024-0", "year": 2023, "title": "Expectations data in asset pricing", "journal": "Handbook of Economic Expectations", "volume": None, "issue": None, "authors": [["Klaus", "Adam", None], ["Stefan", "Nagel", None]], "working_paper_institution": None}, {"reference_type": "working_paper", "doi": "10.3386/w31856", "year": None, "title": "Who Invests in Crypto? Wealth, Financial Constraints, and Risk Attitudes", "journal": None, "volume": None, "issue": None, "authors": [["Darren", "Aiello", None], ["Scott", "Baker", None], ["Tetyana", "Balyuk", None], ["Marco Di", "Maggio", None], ["Mark", "Johnson", None], ["Jason", "Kotter", None]], "working_paper_institution": None}, {"title": "Regulating cryptocurrencies: Assessing market reactions", "author": "Auer", "year": "2018", "journal": "BIS Q. Rev. Sept.", "reference_type": "article"}, {"reference_type": "article", "doi": "10.1111/0022-1082.00226", "year": 2000, "title": "Trading Is Hazardous to Your Wealth: The Common Stock Investment Performance of Individual Investors", "journal": "The Journal of Finance", "volume": "55", "issue": "2", "authors": [["Brad M.", "Barber", None], ["Terrance", "Odean", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/j.jfineco.2018.04.007", "year": 2018, "title": "Extrapolation and bubbles", "journal": "Journal of Financial Economics", "volume": "129", "issue": "2", "authors": [["Nicholas", "Barberis", None], ["Robin", "Greenwood", None], ["Lawrence", "Jin", None], ["Andrei", "Shleifer", None]], "working_paper_institution": None}]
        }
    
    # Run in appropriate mode
    if args.mode == 'evaluate':
        results = evaluate_with_held_out(
            paper_json, 
            G, 
            metadata, 
            model, 
            node_features, 
            node_mapping, 
            reverse_mapping, 
            author_coauthor_G,
            observed_ratio=args.observed_ratio,
            top_k=args.top_k,
            device=args.device
        )
    else:
        results = recommend_for_paper(
            paper_json, 
            G, 
            metadata, 
            model, 
            node_features, 
            node_mapping, 
            reverse_mapping, 
            author_coauthor_G,
            top_k=args.top_k,
            device=args.device
        )
    
    # Print results
    if 'error' in results:
        print(f"Error: {results['error']}")
    else:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"{rec['rank']}. {rec['title']} (Score: {rec['score']:.4f})")
            print(f"   Authors: {rec['authors']}")
            print(f"   Journal: {rec['journal']}")
        
        if args.mode == 'evaluate':
            print(f"\nEvaluation Metrics:")
            print(f"Hits: {results['hits']} / {results['held_out_citations']}")
            print(f"Recall@{args.top_k}: {results[f'recall@{args.top_k}']:.4f}")
            print(f"Precision@{args.top_k}: {results[f'precision@{args.top_k}']:.4f}")
            print(f"NDCG@{args.top_k}: {results[f'ndcg@{args.top_k}']:.4f}")
    
    # Save results to output file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 
