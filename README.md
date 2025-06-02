# Citation Recommendation System

This repository contains a citation recommendation system that uses a Graph Neural Network (GNN) to suggest additional citations for academic papers based on partial citation information.

## Overview

The system works by:
1. Training a GNN on a citation network to learn node embeddings
2. Using an MLP to predict citation relationships based on node embeddings and edge features
3. Recommending additional citations for papers based on their existing citations

The model is particularly useful for suggesting relevant citations that might have been missed by the authors.

## Files

- `create_gnn_graphs.py`: Creates citation graphs from article data
- `gnn_model.py`: Implements the GNN model for citation prediction
- `evaluate_gnn.py`: Evaluates the model and recommends citations
- `test_citation_recommendation.py`: Example script for using the system

## Prerequisites

- Python 3.7+
- PyTorch
- PyTorch Geometric
- NetworkX
- NumPy
- tqdm
- scikit-learn
- matplotlib

## Installation

```bash
pip install torch torch_geometric networkx numpy tqdm scikit-learn matplotlib
```

## Usage

### 1. Training the Model

First, you need to train the model using your article data:

```bash
python gnn_model.py
```

This will:
- Load articles from `articles.jsonl`
- Create citation graphs
- Train the GNN model
- Save the model and data to the `cache_gnn` directory

### 2. Recommending Citations

Once the model is trained, you can use it to recommend citations:

```bash
python evaluate_gnn.py --input paper.json --output recommendations.json --mode recommend --top-k 10
```

Options:
- `--input`: Path to a JSON file containing paper information
- `--output`: Path to save the recommendations
- `--mode`: `recommend` (use all citations) or `evaluate` (hold out some citations)
- `--top-k`: Number of top recommendations to return
- `--observed-ratio`: Ratio of citations to observe (for evaluation mode)
- `--device`: `cpu` or `cuda`

### 3. Running the Test Script

For a quick test with an example paper:

```bash
python test_citation_recommendation.py --output-dir results --observed-ratio 0.75 --top-k 10
```

## Input Format

The input paper should be in JSON format:

```json
{
  "title": "Paper Title",
  "published_date": "2024-01-01",
  "authors": [["First", "Last", null]],
  "references": [
    {
      "reference_type": "article",
      "title": "Referenced Paper Title",
      "authors": [["First", "Last", null]],
      "year": 2020
    }
  ]
}
```

## Output Format

The output is a JSON file containing:

```json
{
  "paper_id": "paper_id",
  "paper_title": "Paper Title",
  "existing_citations": 5,
  "recommendations": [
    {
      "rank": 1,
      "id": "paper_id",
      "title": "Recommended Paper Title",
      "authors": "Author Names",
      "journal": "Journal Name",
      "score": 0.95
    }
  ]
}
```

## Evaluation Mode

In evaluation mode, the system:
1. Takes a portion of the paper's citations as observed
2. Holds out the remaining citations
3. Recommends citations based on observed citations
4. Evaluates recommendations against held-out citations

Metrics include:
- Recall@k: Proportion of held-out citations in top k recommendations
- Precision@k: Proportion of top k recommendations that are correct
- NDCG@k: Normalized Discounted Cumulative Gain

## Notes

- The model works best when the paper already has some citations
- The quality of recommendations depends on the coverage of the citation network
- Papers not in the training data cannot be recommended

## To do

- Explain why a given recommendation was made (i.e. strongest edge feature or node embedding similarity)
