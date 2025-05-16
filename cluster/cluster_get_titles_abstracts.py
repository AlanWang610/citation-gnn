import json
import re
import pandas as pd
import os
import unicodedata
from collections import defaultdict
from itertools import product

def should_filter_title(title):
    """Check if title should be filtered out."""
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
        
    title = title.lower()
    return any(filter_term in title for filter_term in titles_to_filter)

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

# Create a dictionary to store unique papers (title -> article)
unique_papers = {}

# Load articles from articles.jsonl
file_path = 'articles.jsonl'
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            
            # Clean HTML tags and their contents from title if it exists
            if 'title' in article:
                article['title'] = re.sub('<[^>]*>.*?</[^>]*>', '', article['title'])
                title = article['title'].strip()
                
                # Skip filtered titles
                if should_filter_title(title):
                    continue
                
                # Clean HTML tags from abstract if it exists
                abstract = ''
                if 'abstract' in article and article['abstract'] is not None:
                    abstract = re.sub('<[^>]*>(.*?)</[^>]*>', r'\1', article['abstract'])
                
                if title and title not in unique_papers:
                    unique_papers[title] = {
                        'title': title,
                        'abstract': abstract.strip(),
                        'is_main_article': True,
                        'has_abstract': abstract.strip() != ''
                    }
                
                # Process references from article
                if 'references' in article:
                    for ref in article['references']:
                        if ref is None:
                            continue
                        
                        if ref.get('reference_type') == 'article':
                            if 'title' not in ref or ref.get('title') is None:
                                continue
                            
                            ref_title = ref.get('title', '').strip()
                            if ref_title and ref_title not in unique_papers:
                                unique_papers[ref_title] = {
                                    'title': ref_title,
                                    'abstract': '',  # References don't have abstracts
                                    'is_main_article': False,
                                    'has_abstract': False
                                }
else:
    print(f"Warning: File not found: {file_path}")

# Convert to DataFrame
papers_df = pd.DataFrame(list(unique_papers.values()))

# Order by priority: main articles with abstracts, main articles without abstracts, references
papers_df['priority'] = 3  # Default priority for references
if not papers_df.empty:
    papers_df.loc[(papers_df['is_main_article']) & (~papers_df['has_abstract']), 'priority'] = 2  # Main articles without abstracts
    papers_df.loc[(papers_df['is_main_article']) & (papers_df['has_abstract']), 'priority'] = 1  # Main articles with abstracts

    # Sort by priority
    papers_df = papers_df.sort_values('priority')

    # Keep only title and abstract columns for the final output
    papers_df_output = papers_df[['title', 'abstract']]

    # Save to pickle file instead of CSV to preserve commas in titles
    papers_df_output.to_pickle('papers_titles_abstracts.pkl')

    # Count each category
    main_with_abstract = papers_df[(papers_df['is_main_article']) & (papers_df['has_abstract'])].shape[0]
    main_without_abstract = papers_df[(papers_df['is_main_article']) & (~papers_df['has_abstract'])].shape[0]
    references = papers_df[~papers_df['is_main_article']].shape[0]

    print(f"Saved {len(papers_df)} unique papers to papers_titles_abstracts.pkl")
    print(f"Main articles with abstracts: {main_with_abstract}")
    print(f"Main articles without abstracts: {main_without_abstract}")
    print(f"Reference articles: {references}")
else:
    print("No papers found to process.")
