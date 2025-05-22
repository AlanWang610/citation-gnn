import json
import pickle as pkl
import random
import re
import os
import unicodedata
from collections import defaultdict
from itertools import product
import networkx as nx
import numpy as np
from tqdm import tqdm

def should_filter_title(title):
    """Check if title should be filtered out."""
    # Handle None or empty titles
    if not title:
        return True  # Filter out entries with no title
        
    titles_to_filter = {
        "editorial board", "publishers note", "advert:", "call for papers:", 
        "table of contents", "forthcoming articles", "cover", "index", "acknowledgments", 
        "erratum", "acknowledgements", "joint editorial", "a note from the editor", "annual report", 
        "oup accepted manuscript", "turnaround times", "recent referees", "front matter", "back matter", 
        "prize announcement", "index to volume", "masthead", "front cover", "back cover", 
        "journal of political economy", "jpe submissions", "annual meeting", "medalist", 
        "executive commmittee meetings", "report of the", "american economic association",
        "errata", "corrigendum", "corrigenda", "editors' introduction", "editor's introduction", "foreword",
        "list of online reports", "minutes", "american economic review",
        "journal of economic literature", "journal of economic perspectives",
        "american economic journal", "job openings for economists", "committee on", "committee for", 
        "general information on the association", "publisher's note", "special issue contents",
        "editorial data", "editor's note", "editorial", "notice", "data on time to first decision",
        "call for papers", "participant schedule", "preliminary program", "american finance association",
        "issue information", "dimensional fund advisors", "brattle", "fischer black prize", "miscellanea", "announcement",
        "referee list", "contents", "subscription page", "title page", "table of content", "note from the editor",
        "backmatter", "frontmatter", "forthcoming papers", "submission of manuscripts", "election of fellows",
        "a comment on", "comments on", "reply to", "referees", "fellows of", "news notes", "nomination of fellows",
        "econometric society", "in memoriam", "correction", "retraction", "presidential address"
    }
    
    if title in ["Index", "Content", "Comment", "Introduction"]:
        return True
        
    title_lower = title.lower()
    return any(filter_term in title_lower for filter_term in titles_to_filter)

def format_authors(authors):
    """Convert authors list to string representation."""
    if not authors:
        return ""
    # Filter out None values from each author list before joining
    return "; ".join(" ".join(item for item in author if item is not None) for author in authors)

def add_period_variants(names):
    """Add variants with periods to journal names."""
    expanded = set()
    for name in names:
        words = name.split()
        for i in range(len(words)):
            variants = [(w, w + '.') for w in words]
            for combo in product(*variants):
                expanded.add(' '.join(combo))
    return expanded

def standardize_journal_name(journal_name):
    """Return standardized journal name."""
    if not journal_name:
        return None
        
    journal_name = journal_name.lower()
    
    # Define journal name sets
    jf_names = {"j finan", "journal finan", "j financ", "journal financ", "j finance", "journal finance", 
                "j of finan", "journal of finan", "j of financ", "journal of financ", "j of finance", 
                "journal of finance"}
    
    jfe_names = {"j finan econ", "journal finan econ", "j financ econ", "journal financ econ", 
                 "j finance econ", "journal finance econ", "j of finan econ", "journal of finan econ", 
                 "j of financ econ", "journal of financ econ", "j of finance econ", 
                 "journal of finance econ", "journal of financial economics"}
    
    rfs_names = {"rev finan stud", "review finan stud", "rev financ stud", "review financ stud", 
                 "rev finance stud", "review finance stud", "rev of finan stud", "review of finan stud", 
                 "rev of financ stud", "review of financ stud", "rev of finance stud", 
                 "review of finance stud", "review of financial studies"}
    
    aer_names = {"am econ rev", "ame econ rev", "amer econ rev", "american econ rev", "am econ review", 
                 "ame econ review", "amer econ review", "american econ review", "american economic review"}
    
    econometrica_names = {"econometrica"}
    
    qje_names = {"q j econ", "q j of econ", "q j economics", "q j of economics", "quart j econ", 
                 "quart j of econ", "quart j economics", "quart j of economics", "quarterly j econ", 
                 "quarterly j of econ", "quarterly j economnics", "quarterly j of economnics", 
                 "quarterly journal economics", "quarterly journal of economics"}
    
    res_names = {"rev econ stud", "review of economics studies", "review of economic studies"}
    
    jpe_names = {"j polit econ", "j polit economics", "j political econ", "j political economics", 
                 "journal of political economy"}
    
    # Add period variants
    jf_names = add_period_variants(jf_names)
    jfe_names = add_period_variants(jfe_names)
    rfs_names = add_period_variants(rfs_names)
    aer_names = add_period_variants(aer_names)
    econometrica_names = add_period_variants(econometrica_names)
    qje_names = add_period_variants(qje_names)
    res_names = add_period_variants(res_names)
    jpe_names = add_period_variants(jpe_names)
    
    # Add 'the' variants
    for name_set in [jf_names, jfe_names, rfs_names, aer_names, econometrica_names, qje_names, res_names, jpe_names]:
        name_set.update({f"the {name}" for name in name_set})
    
    # Journal name mapping
    if any(name in journal_name for name in jfe_names):
        return "Journal of Financial Economics"
    elif any(name in journal_name for name in jf_names):
        return "Journal of Finance"
    elif any(name in journal_name for name in rfs_names):
        return "Review of Financial Studies"
    elif any(name in journal_name for name in aer_names):
        return "American Economic Review"
    elif any(name in journal_name for name in econometrica_names):
        return "Econometrica"
    elif any(name in journal_name for name in qje_names):
        return "Quarterly Journal of Economics"
    elif any(name in journal_name for name in res_names):
        return "Review of Economic Studies"
    elif any(name in journal_name for name in jpe_names):
        return "Journal of Political Economy"
    
    return None

def normalize_name(name):
    """Remove accents and diacritics from author names."""
    return ''.join(c for c in unicodedata.normalize('NFKD', name)
                  if not unicodedata.combining(c))

def get_paper_id(title, authors, year):
    """Create unique paper ID from title, authors and year."""
    if not title or not year:
        return None
        
    # Get sorted list of author last names
    author_names = []
    if authors:
        for author in authors:
            if len(author) >= 2:  # Check if author has first and last name
                author_names.append(author[1].lower())  # Get last name (index 1)
    author_names.sort()
    
    # Create ID string
    author_str = '_'.join(author_names) if author_names else 'unknown'
    title_str = title.lower().replace(' ', '_')
    return f"{title_str}_{author_str}_{year}"

def format_author_name(author):
    """Format author name as first initial and last name."""
    try:
        # If author is not a list or doesn't have at least 2 elements, return None
        if not isinstance(author, list) or len(author) < 2:
            return None
            
        first_name = author[0]
        last_name = author[1]
        
        # Handle case where first name is empty but last name exists
        if not first_name and last_name:
            # Try to extract initial from last name if it contains multiple words
            parts = last_name.split()
            if len(parts) > 1:
                # Assume format is "FirstName LastName" in the last_name field
                first_initial = parts[0][0].lower() if parts[0] else ''
                actual_last_name = ' '.join(parts[1:]).lower()
                
                # Normalize names to remove accents and diacritics
                first_initial = normalize_name(first_initial)
                actual_last_name = normalize_name(actual_last_name)
                
                if first_initial:
                    return f"{first_initial}. {actual_last_name}"
            
            # If we can't extract a first initial, use the first character of the last name
            # This is a fallback for cases like ['', 'Rodriguez-Clare A.', None]
            if '.' in last_name:
                # Try to find an initial pattern like "LastName A."
                parts = last_name.split()
                if len(parts[-1]) == 2 and parts[-1][-1] == '.':
                    first_initial = parts[-1][0].lower()
                    actual_last_name = ' '.join(parts[:-1]).lower()
                    
                    # Normalize names
                    first_initial = normalize_name(first_initial)
                    actual_last_name = normalize_name(actual_last_name)
                    
                    if first_initial:
                        return f"{first_initial}. {actual_last_name}"
            
            # Last resort: use 'x' as a placeholder initial
            return f"x. {normalize_name(last_name.lower())}"
        
        # Normal case: first name and last name both exist
        if first_name and last_name:
            first_initial = first_name[0].lower()
            last_name = last_name.lower()
            
            # Normalize names
            first_initial = normalize_name(first_initial)
            last_name = normalize_name(last_name)
            
            if first_initial:
                return f"{first_initial}. {last_name}"
        
        return None
    except Exception as e:
        return None

def get_journal_encoding(journal_name):
    """
    Create a one-hot encoding for journal names.
    
    Args:
        journal_name: Name of the journal to encode
        
    Returns:
        List of integers representing the journal encoding
    """
    # List of journals we want to encode
    journals = [
        "Journal of Finance",
        "Journal of Financial Economics",
        "Review of Financial Studies",
        "American Economic Review",
        "Econometrica",
        "Quarterly Journal of Economics",
        "Review of Economic Studies",
        "Journal of Political Economy"
    ]
    
    # Initialize encoding with zeros
    encoding = [0] * len(journals)
    
    # Set the corresponding journal's position to 1
    if journal_name:
        standardized_name = standardize_journal_name(journal_name)
        if standardized_name in journals:
            encoding[journals.index(standardized_name)] = 1
    
    return encoding

def load_paper_embeddings(embedding_file):
    """
    Load paper embeddings from a pickle file.
    
    Args:
        embedding_file: Path to the pickle file containing embeddings
        
    Returns:
        Dictionary mapping paper titles to embeddings
    """
    try:
        print(f"Loading paper embeddings from {embedding_file}...")
        papers_with_embeddings = pkl.load(open(embedding_file, 'rb'))
        
        # Create a dictionary to map paper titles to their embeddings
        paper_embeddings_dict = {}
        for _, row in papers_with_embeddings.iterrows():
            if 'title' in row and 'embedding_pca' in row:
                # Convert to Python list to avoid numpy types
                paper_embeddings_dict[row['title']] = [float(x) for x in row['embedding_pca']]
        
        print(f"Loaded embeddings for {len(paper_embeddings_dict)} papers")
        return paper_embeddings_dict
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return {}

def process_articles_to_gnn_graph(articles, embedding_file='embeddings/all_papers_with_embeddings.pkl'):
    """
    Process articles and create a citation network graph for GNN.
    
    Args:
        articles: List of articles to process
        embedding_file: Path to the file containing paper embeddings
        
    Returns:
        Tuple of (graph, metadata_dict, main_article_count)
    """
    # Load paper embeddings
    paper_embeddings_dict = load_paper_embeddings(embedding_file)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Dictionary to store metadata separately
    metadata = {}
    
    # Counter for main articles
    main_article_count = 0
    
    # Dictionary to store paper embeddings
    reduced_paper_embeddings = {}
    
    # First, process all articles to get embeddings
    print("Processing articles to extract embeddings...")
    for article in tqdm(articles):
        # Skip if missing key fields
        if not article.get('title') or not article.get('published_date') or not article.get('authors'):
            continue
            
        # Skip articles that should be filtered out
        if should_filter_title(article.get('title', '')):
            continue
            
        # Get article metadata
        title = article['title']
        year = article['published_date'].split('-')[0]
        authors = article.get('authors', [])
        
        # Create unique ID for article
        article_id = get_paper_id(title, authors, year)
        if not article_id:
            continue
        
        # Get embedding from dictionary if available, otherwise use a default embedding
        if title in paper_embeddings_dict:
            reduced_paper_embeddings[article_id] = paper_embeddings_dict[title]
        else:
            # Use a default embedding if not found
            reduced_paper_embeddings[article_id] = [0.0] * 128  # Assuming 128-dimensional embeddings
        
        # Process references to get their embeddings too
        if 'references' in article:
            for ref in article['references']:
                if ref.get('reference_type') != 'article':
                    continue
                    
                ref_title = ref.get('title')
                if not ref_title:
                    continue
                    
                ref_id = get_paper_id(
                    ref_title,
                    ref.get('authors', []),
                    str(ref.get('year', ''))
                )
                
                if ref_id and ref_id not in reduced_paper_embeddings:
                    # Get embedding for reference if available
                    if ref_title in paper_embeddings_dict:
                        reduced_paper_embeddings[ref_id] = paper_embeddings_dict[ref_title]
                    else:
                        # Use a default embedding if not found
                        reduced_paper_embeddings[ref_id] = [0.0] * 128  # Assuming 128-dimensional embeddings
    
    print(f"Created embeddings for {len(reduced_paper_embeddings)} papers")
    
    # Now build the graph with all necessary node features
    print("Building citation graph with node features...")
    for article in tqdm(articles):
        # Skip if missing key fields
        if not article.get('title') or not article.get('published_date') or not article.get('authors'):
            continue
            
        # Skip articles that should be filtered out
        if should_filter_title(article.get('title', '')):
            continue
            
        # Get article metadata
        title = article['title']
        year = article['published_date'].split('-')[0]
        authors = article.get('authors', [])
        
        # Create unique ID for article
        article_id = get_paper_id(title, authors, year)
        if not article_id:
            continue
        
        # Format authors as first initial and last name
        formatted_authors = []
        for author in authors:
            formatted_name = format_author_name(author)
            if formatted_name:
                formatted_authors.append(formatted_name)
        
        # Get journal encoding
        journal_name = standardize_journal_name(article.get('journal', ''))
        journal_encoding = get_journal_encoding(journal_name)
        
        # Add node for article with features needed for GNN
        G.add_node(article_id, 
                   year=int(year) if year.isdigit() else 0,  # Store year as integer
                   journal_encoding=journal_encoding,  # Store journal encoding
                   scibert_embedding=reduced_paper_embeddings.get(article_id, []))  # Store embedding
        
        # Store additional metadata in separate dictionary
        metadata[article_id] = {
            'title': title,
            'journal': journal_name or '',
            'authors': "; ".join(formatted_authors),
            'is_main_article': True  # Mark as main article
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
                    # Format reference authors
                    ref_formatted_authors = []
                    ref_authors = ref.get('authors', [])
                    for author in ref_authors:
                        formatted_name = format_author_name(author)
                        if formatted_name:
                            ref_formatted_authors.append(formatted_name)
                    
                    # Get journal encoding for reference
                    ref_journal_name = standardize_journal_name(ref.get('journal', ''))
                    ref_journal_encoding = get_journal_encoding(ref_journal_name)
                    
                    # Get reference year
                    ref_year = ref.get('year', '')
                    ref_year_int = int(ref_year) if str(ref_year).isdigit() else 0
                    
                    # Check if node already exists
                    if ref_id not in G:
                        # Add node for reference with features needed for GNN
                        G.add_node(ref_id,
                                  year=ref_year_int,  # Store year as integer
                                  journal_encoding=ref_journal_encoding,  # Store journal encoding
                                  scibert_embedding=reduced_paper_embeddings.get(ref_id, []))  # Store embedding
                        
                        # Store additional metadata in separate dictionary
                        metadata[ref_id] = {
                            'title': ref_title,
                            'journal': ref_journal_name or '',
                            'authors': "; ".join(ref_formatted_authors),
                            'is_main_article': False  # Mark as reference article
                        }
                    
                    # Add edge from main article to reference
                    G.add_edge(article_id, ref_id)
    
    print(f"Created citation network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Processed {main_article_count} main articles")
    
    return G, metadata, main_article_count

def save_gnn_graph(G, metadata, output_dir='cache_gnn'):
    """
    Save the GNN graph and metadata to files.
    
    Args:
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        output_dir: Directory to save files
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the graph - using pickle instead of write_gpickle
    with open(os.path.join(output_dir, 'gnn_citation_graph.gpickle'), 'wb') as f:
        pkl.dump(G, f)
    
    # Save the metadata
    with open(os.path.join(output_dir, 'gnn_paper_metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    print(f"Saved GNN graph and metadata to {output_dir}")

# Function to check graph statistics
def print_graph_statistics(G, metadata):
    """
    Print statistics about the graph.
    
    Args:
        G: NetworkX graph
        metadata: Dictionary of paper metadata
        
    Returns:
        None
    """
    print("\nGraph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Count main articles
    main_articles = [node for node in G.nodes() if metadata[node].get('is_main_article', False)]
    print(f"Number of main articles: {len(main_articles)}")
    
    # Count reference articles
    reference_articles = [node for node in G.nodes() if not metadata[node].get('is_main_article', False)]
    print(f"Number of reference articles: {len(reference_articles)}")
    
    # Check node features
    sample_node = list(G.nodes())[0]
    print("\nSample node features:")
    print(f"Node ID: {sample_node}")
    print(f"Year: {G.nodes[sample_node].get('year')}")
    print(f"Journal encoding: {G.nodes[sample_node].get('journal_encoding')}")
    print(f"SciBERT embedding shape: {len(G.nodes[sample_node].get('scibert_embedding', []))}")
    
    # Check metadata
    print("\nSample node metadata:")
    print(f"Title: {metadata[sample_node].get('title')}")
    print(f"Journal: {metadata[sample_node].get('journal')}")
    print(f"Authors: {metadata[sample_node].get('authors')}")
    print(f"Is main article: {metadata[sample_node].get('is_main_article')}")
    
    # Check citation statistics
    out_degrees = [G.out_degree(node) for node in main_articles]
    if out_degrees:
        print(f"\nCitation statistics for main articles:")
        print(f"Average number of citations: {sum(out_degrees) / len(out_degrees):.2f}")
        print(f"Maximum number of citations: {max(out_degrees)}")
        print(f"Minimum number of citations: {min(out_degrees)}")
    
    # Check in-degree statistics for reference articles
    in_degrees = [G.in_degree(node) for node in reference_articles]
    if in_degrees:
        print(f"\nCitation statistics for reference articles:")
        print(f"Average number of times cited: {sum(in_degrees) / len(in_degrees):.2f}")
        print(f"Maximum number of times cited: {max(in_degrees)}")
        print(f"Minimum number of times cited: {min(in_degrees)}")

# Main function to run the graph creation process
def create_gnn_graphs(articles_file='articles.jsonl', embedding_file='embeddings/all_papers_with_embeddings.pkl'):
    """
    Main function to create GNN graphs from articles.
    
    Args:
        articles_file: Path to the articles file
        embedding_file: Path to the embeddings file
        
    Returns:
        None
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
    
    # Process articles to create GNN graph
    gnn_graph_cache_path = 'cache_gnn/gnn_citation_graph.gpickle'
    gnn_metadata_cache_path = 'cache_gnn/gnn_paper_metadata.json'
    
    if os.path.exists(gnn_graph_cache_path) and os.path.exists(gnn_metadata_cache_path):
        print("Loading GNN graph and metadata from cache...")
        G = nx.read_gpickle(gnn_graph_cache_path)
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
