import networkx as nx
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
from itertools import product
from collections import defaultdict

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
    return "; ".join(" ".join(author) for author in authors)

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

def process_articles_to_graph(articles):
    """Process articles and create a citation network graph."""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Counter for main articles
    main_article_count = 0
    
    # First, add all articles from the input file as main articles
    for article in articles:
        # Skip if missing key fields
        if not article.get('title') or not article.get('published_date') or not article.get('authors'):
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
        
        # Add node for article - explicitly mark as main article
        G.add_node(article_id, 
                   title=title,
                   year=year,
                   journal=standardize_journal_name(article.get('journal', '')) or '',
                   authors="; ".join(formatted_authors),
                   is_main_article=True)  # Always mark as main article
        
        main_article_count += 1
        
        # Process references
        if 'references' in article:
            for ref in article['references']:
                if ref.get('reference_type') != 'article':
                    continue
                    
                ref_id = get_paper_id(
                    ref.get('title'),
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
                    
                    # Check if this reference already exists in the graph
                    if ref_id not in G:
                        # Add node for reference - mark as not a main article
                        G.add_node(ref_id,
                                  title=ref.get('title', ''),
                                  year=str(ref.get('year', '')),
                                  journal=standardize_journal_name(ref.get('journal', '')) or '',
                                  authors="; ".join(ref_formatted_authors),
                                  is_main_article=False)  # Mark as reference article
                    
                    # Add edge from main article to reference
                    G.add_edge(article_id, ref_id)
    
    return G, main_article_count

def process_articles_to_author_graphs(articles):
    """
    Create author coauthorship and citation graphs from articles.
    
    Args:
        articles: List of articles to process
        
    Returns:
        Tuple of (coauthorship_graph, citation_graph, main_author_count)
    """
    # Create graphs for authors
    author_citation_G = nx.DiGraph()  # Directed graph for citations
    author_coauthor_G = nx.Graph()    # Undirected graph for coauthorships
    
    # Dictionary to track citation counts between authors
    citation_counts = defaultdict(lambda: defaultdict(int))
    # Dictionary to track coauthorship counts
    coauthor_counts = defaultdict(lambda: defaultdict(int))
    # Set to track main authors (authors who wrote main papers)
    main_authors = set()
    # Set to track all authors (main + referenced)
    all_authors = set()
    
    # Statistics counters
    total_articles = 0
    articles_with_authors = 0
    total_authors_processed = 0
    valid_authors_count = 0
    empty_first_name_count = 0
    total_references = 0
    references_with_authors = 0
    
    # Process each article directly
    print(f"Processing {len(articles)} articles for author graphs...")
    
    # First pass: Process main articles and their authors
    for article in articles:
        total_articles += 1
        
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
        
        # Skip if authors is not a list
        if not isinstance(authors, list):
            continue
            
        # Format authors
        formatted_authors = []
        for author in authors:
            total_authors_processed += 1
            
            # Check for empty first name
            if isinstance(author, list) and len(author) >= 2 and not author[0] and author[1]:
                empty_first_name_count += 1
            
            formatted_name = format_author_name(author)
            if formatted_name:
                formatted_authors.append(formatted_name)
                main_authors.add(formatted_name)
                all_authors.add(formatted_name)
                valid_authors_count += 1
        
        # Skip if no valid authors
        if not formatted_authors:
            continue
            
        articles_with_authors += 1
        
        # Process coauthorship relationships for main article authors
        for i, author1 in enumerate(formatted_authors):
            for author2 in formatted_authors[i+1:]:
                coauthor_counts[author1][author2] += 1
                coauthor_counts[author2][author1] += 1
    
    # Second pass: Process references and build citation relationships
    for article in articles:
        # Skip if missing key fields
        if not article.get('title') or not article.get('published_date') or not article.get('authors'):
            continue
            
        # Skip articles that should be filtered out
        if should_filter_title(article.get('title', '')):
            continue
            
        # Get main article authors
        main_article_authors = []
        for author in article.get('authors', []):
            formatted_name = format_author_name(author)
            if formatted_name:
                main_article_authors.append(formatted_name)
        
        # Skip if no valid authors for main article
        if not main_article_authors:
            continue
        
        # Process references
        if 'references' in article:
            for ref in article['references']:
                total_references += 1
                
                # Skip references that are not articles or have no title
                if ref.get('reference_type') != 'article' or not ref.get('title'):
                    continue
                
                # Skip references that should be filtered out
                if should_filter_title(ref.get('title', '')):
                    continue
                    
                ref_authors = ref.get('authors', [])
                ref_formatted_authors = []
                
                for ref_author in ref_authors:
                    ref_formatted_name = format_author_name(ref_author)
                    if ref_formatted_name:
                        ref_formatted_authors.append(ref_formatted_name)
                        all_authors.add(ref_formatted_name)
                
                # Skip if no valid authors for reference
                if not ref_formatted_authors:
                    continue
                    
                references_with_authors += 1
                
                # Process coauthorship relationships for reference authors
                for i, author1 in enumerate(ref_formatted_authors):
                    for author2 in ref_formatted_authors[i+1:]:
                        coauthor_counts[author1][author2] += 1
                        coauthor_counts[author2][author1] += 1
                
                # Add citation relationships between main article authors and reference authors
                for main_author in main_article_authors:
                    for ref_author in ref_formatted_authors:
                        if main_author != ref_author:  # Skip self-citations
                            citation_counts[main_author][ref_author] += 1
    
    # Build the citation graph (directed)
    for citing_author, cited_authors in citation_counts.items():
        for cited_author, citation_weight in cited_authors.items():
            author_citation_G.add_edge(citing_author, cited_author, weight=citation_weight)
    
    # Build the coauthorship graph (undirected)
    for author1, coauthors in coauthor_counts.items():
        for author2, coauthor_weight in coauthors.items():
            author_coauthor_G.add_edge(author1, author2, weight=coauthor_weight)
    
    # Print detailed statistics
    print(f"\nAuthor Graph Statistics:")
    print(f"Total articles processed: {total_articles}")
    print(f"Articles with at least one valid author: {articles_with_authors}")
    print(f"Total references processed: {total_references}")
    print(f"References with at least one valid author: {references_with_authors}")
    print(f"Total authors processed: {total_authors_processed}")
    print(f"Valid authors extracted: {valid_authors_count}")
    print(f"Authors with empty first name: {empty_first_name_count}")
    print(f"Unique main authors: {len(main_authors)}")
    print(f"Total unique authors (main + referenced): {len(all_authors)}")
    print(f"Coauthorship edges: {len(author_coauthor_G.edges())}")
    print(f"Citation edges: {len(author_citation_G.edges())}")
    
    return author_coauthor_G, author_citation_G, len(main_authors)
