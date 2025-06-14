#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate GraphSAGE model for citation recommendation.

This script loads a trained GraphSAGE model and recommends additional citations
for a paper with partial citation information.

The script supports both transductive and inductive settings:
- Transductive: The paper is already in the graph
- Inductive: The paper is new and not part of the training data
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
    GraphSAGECitationModel, extract_edge_features, 
    generate_node_features_for_new_paper, inductive_recommend_citations,
    inductive_evaluate_with_held_out
)

def load_model_and_data(cache_dir='cache_gnn'):
    """
    Load the trained GraphSAGE model and graph data from cache.
    If cache files are missing, regenerate them.
    
    Args:
        cache_dir: Directory containing cached files
        
    Returns:
        Tuple of (model, graph, metadata, data, paper_embeddings_dict)
    """
    print("Loading GraphSAGE model and data from cache...")
    
    # Define cache paths
    gnn_graph_path = os.path.join(cache_dir, 'gnn_citation_graph.gpickle')
    gnn_metadata_path = os.path.join(cache_dir, 'gnn_paper_metadata.json')
    gnn_data_path = os.path.join(cache_dir, 'gnn_data.pt')
    model_path = os.path.join(cache_dir, 'graphsage_citation_model.pt')
    embeddings_path = os.path.join(cache_dir, 'paper_embeddings_dict.pkl')
    citation_data_path = os.path.join(cache_dir, 'citation_prediction_data.pt')
    
    # Check if all required files exist
    required_files = [gnn_graph_path, gnn_metadata_path, gnn_data_path, model_path, embeddings_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Missing cache files: {missing_files}")
        print("Regenerating cache files...")
        
        # Import the main function from gnn_model to regenerate cache
        from gnn_model import main as gnn_main
        
        # Run the main function to regenerate all cache files
        gnn_main()
        
        # Check again if files exist after regeneration
        still_missing = [f for f in required_files if not os.path.exists(f)]
        if still_missing:
            raise FileNotFoundError(f"Failed to regenerate required files: {still_missing}")
        
        print("Successfully regenerated cache files")
    
    # Load graph and metadata
    print("Loading citation graph...")
    with open(gnn_graph_path, 'rb') as f:
        G = pkl.load(f)
    
    print("Loading paper metadata...")
    with open(gnn_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load paper embeddings dictionary
    print("Loading paper embeddings...")
    with open(embeddings_path, 'rb') as f:
        paper_embeddings_dict = pkl.load(f)
    
    # Load PyTorch Geometric data
    print("Loading PyTorch Geometric data...")
    data = torch.load(gnn_data_path, weights_only=False)
    
    # Load citation training data to get dimensions
    citation_data = torch.load(citation_data_path, weights_only=False)
    
    # Load model
    print("Loading trained GraphSAGE model...")
    input_dim = data.num_node_features
    edge_feature_dim = citation_data['train']['edge_features'].shape[1]
    
    # Initialize and load model
    model = GraphSAGECitationModel(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        edge_feature_dim=edge_feature_dim,
        dropout=0.2
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Successfully loaded GraphSAGE model and data")
    
    return model, G, metadata, data, paper_embeddings_dict

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

def recommend_citations(paper_id, observed_citations, G, metadata, model, data, 
                       paper_embeddings_dict, top_k=10, exclude_observed=True, 
                       device='cpu', include_feature_importance=False):
    """
    Recommend citations for a paper based on partial citation information.
    
    Args:
        paper_id: ID of the paper
        observed_citations: List of observed citation IDs
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        model: Trained GraphSAGE model
        data: PyTorch Geometric Data object
        paper_embeddings_dict: Dictionary of paper embeddings
        top_k: Number of top recommendations to return
        exclude_observed: Whether to exclude observed citations from recommendations
        device: Device to run the model on
        include_feature_importance: Whether to include feature importance analysis
        
    Returns:
        List of dictionaries containing recommended citations
    """
    # Check if paper exists in graph
    if paper_id not in G:
        print(f"Error: Paper ID {paper_id} not found in graph")
        return []
    
    # Move model and data to device
    model = model.to(device)
    data = data.to(device)
    
    # Get source node index
    src_idx = data.node_mapping[paper_id]
    
    # Get candidate papers (all papers except the source paper)
    candidates = [node for node in G.nodes() if node != paper_id]
    
    # Exclude observed citations if requested
    if exclude_observed:
        candidates = [c for c in candidates if c not in observed_citations]
    
    # Prepare batches for prediction
    batch_size = 1000
    all_probs = []
    all_edge_features = []
    
    # Process candidates in batches
    for i in range(0, len(candidates), batch_size):
        batch_candidates = candidates[i:i+batch_size]
        batch_tgt_idxs = [data.node_mapping[c] for c in batch_candidates]
        
        # Extract edge features for each candidate
        edge_features = []
        for candidate in batch_candidates:
            features = extract_edge_features(
                paper_id, 
                candidate, 
                G, 
                metadata, 
                paper_embeddings_dict,
                observed_citations_only=True,
                observed_citations=observed_citations
            )
            edge_features.append(list(features.values()))
        
        all_edge_features.extend(edge_features)
        
        # Create tensors
        src_tensor = torch.tensor([src_idx] * len(batch_candidates), dtype=torch.long).to(device)
        tgt_tensor = torch.tensor(batch_tgt_idxs, dtype=torch.long).to(device)
        edge_tensor = torch.tensor(edge_features, dtype=torch.float).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(data.x, data.edge_index, src_tensor, tgt_tensor, edge_tensor)
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

def evaluate_with_held_out(paper_json, G, metadata, model, data, paper_embeddings_dict,
                          observed_ratio=0.75, top_k=10, device='cpu', 
                          include_feature_importance=False, random_seed=111):
    """
    Evaluate citation recommendations against held-out citations.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        model: Trained GraphSAGE model
        data: PyTorch Geometric Data object
        paper_embeddings_dict: Dictionary of paper embeddings
        observed_ratio: Ratio of citations to observe
        top_k: Number of top recommendations to consider
        device: Device to run the model on
        include_feature_importance: Whether to include feature importance analysis
        random_seed: Random seed for reproducible citation splitting
        
    Returns:
        Dictionary of evaluation metrics and recommendations
    """
    # Parse paper
    paper_id, all_citations, paper_data = parse_paper(paper_json, G, metadata)
    
    if paper_id is None:
        return {'error': 'Paper not found in graph'}
    
    if len(all_citations) < 2:
        return {'error': 'Paper has too few citations for evaluation'}
    
    # Set random seed for reproducible results
    random.seed(random_seed)
    
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
        data,
        paper_embeddings_dict,
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

def recommend_for_paper(paper_json, G, metadata, model, data, paper_embeddings_dict,
                       top_k=10, device='cpu', include_feature_importance=False):
    """
    Recommend citations for a paper based on its existing citations.
    
    Args:
        paper_json: Dictionary containing paper information
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        model: Trained GraphSAGE model
        data: PyTorch Geometric Data object
        paper_embeddings_dict: Dictionary of paper embeddings
        top_k: Number of top recommendations to return
        device: Device to run the model on
        include_feature_importance: Whether to include feature importance analysis
        
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
        data,
        paper_embeddings_dict,
        top_k=top_k,
        exclude_observed=True,
        device=device,
        include_feature_importance=include_feature_importance
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
    parser = argparse.ArgumentParser(description='Evaluate GraphSAGE model for citation recommendation')
    parser.add_argument('--input', type=str, help='Path to input JSON file containing paper information')
    parser.add_argument('--output', type=str, default='recommendations.json', help='Path to output JSON file')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top recommendations to return')
    parser.add_argument('--mode', type=str, default='recommend', choices=['recommend', 'evaluate'], 
                        help='Mode: recommend (use all citations) or evaluate (hold out some citations)')
    parser.add_argument('--observed-ratio', type=float, default=0.75, 
                        help='Ratio of citations to observe in evaluation mode')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Device to run the model on')
    parser.add_argument('--inductive', action='store_true', 
                        help='Use inductive mode for new papers not in the graph')
    parser.add_argument('--feature-importance', action='store_true',
                        help='Include detailed feature importance analysis for each recommendation')
    parser.add_argument('--random-seed', type=int, default=111,
                        help='Random seed for reproducible evaluation results (default: 111)')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Load model and data
    model, G, metadata, data, paper_embeddings_dict = load_model_and_data()
    
    # Load input paper
    if args.input:
        with open(args.input, 'r') as f:
            paper_json = json.load(f)
    else:
        # Use example paper
        paper_json = {
            "doi": "10.1111/jofi.13455", 
            "type": "journal-article", 
            "published_date": "2025-04-01", 
            "title": "Women in Charge: Evidence from Hospitals", 
            "journal": "The Journal of Finance", 
            "abstract": "<jats:title>ABSTRACT</jats:title><jats:p>The paper examines the decision‐making, compensation, and turnover of female CEOs in U.S. hospitals. Contrary to the literature on lower‐ranked executives and directors in public firms, there is no evidence that gender differences in preferences for risk or altruism affect decision‐making of hospital CEOs: corporate policies do not shift when women take (or leave) office, and male and female CEOs respond similarly to a major financial shock. However, female CEOs earn lower salaries, face flatter pay‐for‐performance incentives, and exhibit greater turnover after poor performance. Hospital boards behave as though they perceive female CEOs as less productive.</jats:p>", 
            "volume": None, 
            "issue": None, 
            "authors": [["KATHARINA", "LEWELLEN", None]], 
            "references": [{"reference_type": "article", "doi": "10.1016/j.jfineco.2008.10.007", "year": 2009, "title": "Women in the boardroom and their impact on governance and performance⁎", "journal": "Journal of Financial Economics", "volume": "94", "issue": "2", "authors": [["Renée B.", "Adams", None], ["Daniel", "Ferreira", None]], "working_paper_institution": None}]
        }
    
    # Check if paper exists in graph
    paper_id = get_paper_id(
        paper_json.get('title', ''),
        paper_json.get('authors', []),
        paper_json.get('published_date', '').split('-')[0]
    )
    
    # Determine if inductive mode is needed
    inductive_mode = args.inductive or (paper_id not in G)
    
    if inductive_mode:
        print("Using inductive mode for citation recommendation")
        
        # Run in appropriate mode
        if args.mode == 'evaluate':
            results = inductive_evaluate_with_held_out(
                paper_json, 
                G, 
                metadata, 
                model, 
                data,
                paper_embeddings_dict,
                observed_ratio=args.observed_ratio,
                top_k=args.top_k,
                device=args.device,
                random_seed=args.random_seed,
                include_feature_importance=args.feature_importance
            )
        else:
            # Recommend mode
            recommendations, paper_id, observed_citations, paper_metadata = inductive_recommend_citations(
                paper_json, 
                G, 
                metadata, 
                model, 
                data,
                paper_embeddings_dict,
                top_k=args.top_k,
                device=args.device,
                include_feature_importance=args.feature_importance
            )
            
            results = {
                'paper_id': paper_id,
                'paper_title': paper_metadata['title'],
                'existing_citations': len(observed_citations),
                'recommendations': recommendations
            }
    else:
        print("Using transductive mode for citation recommendation")
        
        # Run in appropriate mode
        if args.mode == 'evaluate':
            results = evaluate_with_held_out(
                paper_json, 
                G, 
                metadata, 
                model, 
                data,
                paper_embeddings_dict,
                observed_ratio=args.observed_ratio,
                top_k=args.top_k,
                device=args.device,
                include_feature_importance=args.feature_importance,
                random_seed=args.random_seed
            )
        else:
            results = recommend_for_paper(
                paper_json, 
                G, 
                metadata, 
                model, 
                data,
                paper_embeddings_dict,
                top_k=args.top_k,
                device=args.device,
                include_feature_importance=args.feature_importance
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
