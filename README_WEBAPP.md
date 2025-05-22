# Citation Recommendation Web Application

This web application provides a user-friendly interface for the citation recommendation system, allowing researchers to find relevant citations for their academic papers.

## Features

- **Paper Information Input**: Enter your paper's title, authors, publication date, journal, and abstract.
- **Reference Management**: Add existing references/citations to improve recommendation quality.
- **Multiple Modes**:
  - **Recommend Mode**: Provides new citation recommendations based on your paper and its existing references.
  - **Evaluate Mode**: Holds out a portion of your citations to evaluate the recommendation algorithm's performance.
- **Two Approaches**:
  - **Inductive**: For new papers not in the training data.
  - **Transductive**: For papers already in the citation graph.
- **Customizable Settings**: Adjust parameters like observation ratio and number of recommendations.
- **Visual Results**: View recommendations with scores, status indicators, and evaluation metrics.

## Installation

1. Ensure you have all the required dependencies:
```bash
pip install flask torch networkx numpy tqdm
```

2. Make sure the model and data files are present in the `cache_gnn` directory.

## Usage

1. Start the web server:
```bash
python app.py
```

2. Open your web browser and navigate to [http://localhost:5000](http://localhost:5000)

3. Use the web interface to:
   - Enter your paper details
   - Add references (optional but recommended)
   - Adjust settings
   - Get citation recommendations

## Example Paper

The system includes a pre-loaded example paper titled "Women in Charge: Evidence from Hospitals" for demonstration purposes. Click the "Load Example Paper" button to populate the form with this example.

## Settings Explained

- **Mode**:
  - **Recommend**: Use all citations to recommend new ones (standard mode).
  - **Evaluate**: Randomly hold out some citations to test how well the system can recover them.

- **Approach**:
  - **Inductive**: Use when your paper is new and not part of the training data.
  - **Transductive**: Use when your paper already exists in the citation graph.

- **Observed Ratio**: In evaluation mode, this determines what percentage of citations are observed (vs. held out).

- **Number of Recommendations**: How many top recommendations to show.

- **Device**: Choose CPU or CUDA (GPU) for processing. GPU is faster but requires compatible hardware.

## Understanding Results

In the results section, you'll see:

- **Evaluation Metrics** (in evaluate mode):
  - **Held Out Citations**: Number of citations held out for evaluation.
  - **Hits**: Number of held-out citations successfully recovered.
  - **Recall@K**: Percentage of held-out citations found in the top K recommendations.
  - **Precision@K**: Percentage of top K recommendations that are actually held-out citations.
  - **NDCG@K**: Normalized Discounted Cumulative Gain, measuring the ranking quality.

- **Recommendation Table**:
  - **Status Indicators**:
    - ✓ (Green): Actual citation that was held out (evaluate mode)
    - × (Red): Not cited in your paper
    - N (Blue): New recommendation
    - C (Gray): Already cited in your paper

  - **Score**: Higher scores indicate stronger recommendations.

## Troubleshooting

- **Long Loading Times**: The first time you use the application, it may take time to load the model and citation graph data.
- **Memory Issues**: The citation recommendation models require significant memory. Ensure your system has at least 8GB of available RAM.
- **Missing Files**: If you encounter errors about missing files, make sure all the required model and data files are present in the `cache_gnn` directory.

## Citation

If you use this system in your research, please cite:

```
@misc{citationGNN,
  author = {Your Name},
  title = {Citation GNN: A Graph Neural Network for Citation Recommendation},
  year = {2023},
  howpublished = {\url{https://github.com/yourusername/citation-gnn}}
}
``` 
