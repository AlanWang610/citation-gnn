import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from tqdm import tqdm
from joblib import Parallel, delayed
import os

# Load the pickle file with titles and abstracts
print("Loading data from pickle file...")
papers_df = pd.read_pickle('papers_titles_abstracts.pkl')

# Load SciBERT model and tokenizer
print("Loading SciBERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

def embed_text(text, tokenizer, model):
    """Get embeddings for text using SciBERT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Get the [CLS] token embedding

# Create a function to combine title and abstract if available
def get_text_to_embed(row):
    if row['abstract'] and not pd.isna(row['abstract']) and row['abstract'].strip():
        return row['title'] + " " + row['abstract']
    else:
        return row['title']

# Function to process a single paper
def process_paper(row_tuple):
    i, row = row_tuple
    text_to_embed = get_text_to_embed(row)
    embedding = embed_text(text_to_embed, tokenizer, model)
    return embedding[0]  # Extract from batch dimension

# Determine number of cores to use
num_cores = os.cpu_count()
print(f"Using {num_cores} CPU cores for parallel processing...")

# Generate embeddings for each paper in parallel
print("Generating embeddings for papers in parallel...")
embeddings = Parallel(n_jobs=num_cores)(
    delayed(process_paper)((i, row)) for i, row in tqdm(
        papers_df.iterrows(), 
        total=len(papers_df),
        desc="Processing papers",
        ncols=100,
        position=0,
        leave=True
    )
)

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings)

# Add embeddings to dataframe
papers_df['embedding'] = list(embeddings_array)

# Apply PCA to reduce dimensions to 128
print("Applying PCA to reduce dimensions to 128...")
pca = PCA(n_components=128)
reduced_embeddings = pca.fit_transform(embeddings_array)

# Add reduced embeddings to dataframe
papers_df['embedding_pca'] = list(reduced_embeddings)

# Save the dataframe with embeddings back to pickle
print("Saving dataframe with embeddings...")
papers_df.to_pickle('papers_with_embeddings.pkl')

print(f"Processed {len(papers_df)} papers. Embeddings shape: {embeddings_array.shape}, PCA shape: {reduced_embeddings.shape}")
