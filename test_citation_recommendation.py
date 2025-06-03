#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for citation recommendation using evaluate_gnn.py.

This script demonstrates how to use the citation recommendation system
with the example paper from the original task.

The script supports both transductive and inductive settings:
- Transductive: The paper is already in the graph
- Inductive: The paper is new and not part of the training data
"""

import json
import os
import argparse
import random

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description='Test citation recommendation system')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--observed-ratio', type=float, default=0.75, 
                        help='Ratio of citations to observe (default: 0.75)')
    parser.add_argument('--top-k', type=int, default=50, help='Number of top recommendations to return (default: 50)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Device to run the model on')
    parser.add_argument('--inductive', action='store_false',
                        help='Use transductive mode (default is inductive)')
    parser.add_argument('--test-paper', type=str, default=None,
                        help='Path to a JSON file containing a test paper (optional)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test paper from example_paper.jsonl
    with open('example_paper.jsonl', 'r') as f:
        example_paper = json.loads(f.read().strip())
    
    print(f"Loaded test paper: {example_paper['title']}")
    print(f"Total references: {len(example_paper.get('references', []))}")
    
    # Save example paper to a file
    example_paper_path = os.path.join(args.output_dir, 'example_paper.json')
    with open(example_paper_path, 'w') as f:
        json.dump(example_paper, f, indent=2)
    
    print(f"Saved example paper to {example_paper_path}")
    
    # Set inductive mode flag
    inductive_flag = "--inductive" if args.inductive else ""
    
    # Run evaluate_gnn.py in evaluation mode
    evaluation_output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    
    # Define mode name for logging
    mode_name = "inductive" if args.inductive else "transductive"
    
    # Evaluation command (removed feature importance for now)
    cmd = f"python evaluate_gnn.py --input {example_paper_path} --output {evaluation_output_path} --mode evaluate --observed-ratio {args.observed_ratio} --top-k {args.top_k} --device {args.device} {inductive_flag}"
    
    print(f"Running {mode_name} evaluation: {cmd}")
    os.system(cmd)
    
    # Process the evaluation results to mark citations
    with open(evaluation_output_path, 'r') as f:
        evaluation_results = json.load(f)
    
    # Check for errors
    if 'error' in evaluation_results:
        print(f"Error in evaluation: {evaluation_results['error']}")
    else:
        # Get held-out citations
        held_out_citations = evaluation_results.get('held_out_citation_ids', [])
        
        # Mark recommendations as actual citations or not in dataset
        for rec in evaluation_results.get('recommendations', []):
            if rec['id'] in held_out_citations:
                rec['status'] = 'ACTUAL_CITATION'
                rec['checkbox'] = '[✓]'  # ASCII-compatible checkbox for actual citations
            else:
                # Check if this paper is in the dataset but not cited
                rec['status'] = 'NOT_CITED'
                rec['checkbox'] = '[ ]'  # ASCII-compatible checkbox for non-citations
        
        # Save the updated results
        with open(evaluation_output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
    
    # Run evaluate_gnn.py in recommendation mode
    recommendation_output_path = os.path.join(args.output_dir, 'recommendation_results.json')
    cmd = f"python evaluate_gnn.py --input {example_paper_path} --output {recommendation_output_path} --mode recommend --top-k {args.top_k} --device {args.device} {inductive_flag}"
    
    print(f"Running {mode_name} recommendation: {cmd}")
    os.system(cmd)
    
    # Process the recommendation results
    try:
        with open(recommendation_output_path, 'r') as f:
            recommendation_results = json.load(f)
        
        # Check for errors
        if 'error' in recommendation_results:
            print(f"Error in recommendation: {recommendation_results['error']}")
        else:
            # Get all citations from the original paper
            all_citations = []
            for ref in example_paper.get('references', []):
                if ref.get('reference_type') == 'article' and ref.get('title'):
                    all_citations.append(ref.get('title'))
            
            # Mark recommendations
            for rec in recommendation_results.get('recommendations', []):
                if any(citation_title.lower() in rec['title'].lower() or rec['title'].lower() in citation_title.lower() 
                       for citation_title in all_citations):
                    rec['status'] = 'ALREADY_CITED'
                    rec['checkbox'] = '[C]'  # ASCII-compatible marker for already cited papers
                else:
                    rec['status'] = 'NEW_RECOMMENDATION'
                    rec['checkbox'] = '[N]'  # ASCII-compatible marker for new recommendations
            
            # Save the updated recommendation results
            with open(recommendation_output_path, 'w') as f:
                json.dump(recommendation_results, f, indent=2)
            
            # Print summary with status information
            print("\nEvaluation Results Summary:")
            if 'error' in evaluation_results:
                print(f"Error: {evaluation_results['error']}")
            else:
                print(f"Total held-out citations: {len(held_out_citations)}")
                actual_citations_found = sum(1 for rec in evaluation_results.get('recommendations', []) if rec.get('status') == 'ACTUAL_CITATION')
                print(f"Actual citations found in top {args.top_k}: {actual_citations_found}")
                
                # Print evaluation metrics
                if 'recall@'+str(args.top_k) in evaluation_results:
                    print(f"Recall@{args.top_k}: {evaluation_results[f'recall@{args.top_k}']:.4f}")
                    print(f"Precision@{args.top_k}: {evaluation_results[f'precision@{args.top_k}']:.4f}")
                    print(f"NDCG@{args.top_k}: {evaluation_results[f'ndcg@{args.top_k}']:.4f}")
                
                # Print ALL evaluation recommendations with status markers
                print(f"\nALL Top {args.top_k} Evaluation Recommendations:")
                print("=" * 100)
                for rec in evaluation_results.get('recommendations', []):
                    status_marker = rec.get('checkbox', '')
                    print(f"{rec.get('rank', 0):2d}. {status_marker} {rec.get('title', 'Unknown Title')} (Score: {rec.get('score', 0):.4f})")
                    print(f"    Authors: {rec.get('authors', '')}")
                    print(f"    Journal: {rec.get('journal', '')}")
                    print(f"    Status: {rec.get('status', '')}")
                    
                    # Show SciBERT cosine similarity if available in edge features
                    print(f"    GraphSAGE + SciBERT Cosine Similarity Model")
                    print()
            
            print("\nRecommendation Results Summary:")
            already_cited = sum(1 for rec in recommendation_results.get('recommendations', []) if rec.get('status') == 'ALREADY_CITED')
            new_recommendations = sum(1 for rec in recommendation_results.get('recommendations', []) if rec.get('status') == 'NEW_RECOMMENDATION')
            print(f"Already cited papers in recommendations: {already_cited}")
            print(f"New recommendations: {new_recommendations}")
            
            # Print ALL recommendation results with status markers
            print(f"\nALL Top {args.top_k} Recommendation Results:")
            print("=" * 100)
            for rec in recommendation_results.get('recommendations', []):
                status_marker = rec.get('checkbox', '')
                print(f"{rec.get('rank', 0):2d}. {status_marker} {rec.get('title', 'Unknown Title')} (Score: {rec.get('score', 0):.4f})")
                print(f"    Authors: {rec.get('authors', '')}")
                print(f"    Journal: {rec.get('journal', '')}")
                print(f"    Status: {rec.get('status', '')}")
                
                # Show model info
                print(f"    🧠 GraphSAGE + SciBERT Cosine Similarity Model")
                print()
    except Exception as e:
        print(f"Error processing recommendation results: {e}")
    
    # Print mode summary
    print(f"\nTest complete for {mode_name} mode!")
    print(f"Evaluation results saved to {evaluation_output_path}")
    print(f"Recommendation results saved to {recommendation_output_path}")
    print(f"Legend: [✓]=Actual citation that was held out, [ ]=Not cited, [C]=Already cited, [N]=New recommendation")
    print(f"🧠 Using GraphSAGE for structural learning + SciBERT cosine similarity for semantic features")
    print(f"   - Node features: Year, journal encoding, number of authors, is_main_article")
    print(f"   - Edge features: SciBERT cosine similarity, shared authors, shared citations, journal match, year diff, citation context")
    print(f"   - GraphSAGE learns structural patterns while SciBERT captures semantic similarity")

if __name__ == "__main__":
    main()
