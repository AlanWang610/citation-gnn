#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask application for citation recommendation.

This application provides a web interface for the citation recommendation system,
allowing users to input paper details and view recommended citations.
"""

import os
import json
import tempfile
from flask import Flask, render_template, request, jsonify, session

from evaluate_gnn import (
    load_model_and_data, evaluate_with_held_out, 
    recommend_for_paper, inductive_evaluate_with_held_out,
    inductive_recommend_citations
)

# Global variables to store model and data
model = None
G = None
metadata = None
node_features = None
node_mapping = None
reverse_mapping = None
author_coauthor_G = None
EXAMPLE_PAPER = None

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    
    # Example paper as defined in test_citation_recommendation.py
    # This will be replaced with the full version in initialize_app()
    app_example_paper = {"doi": "10.1111/jofi.13455", "type": "journal-article", "published_date": "2025-04-01", "title": "Women in Charge: Evidence from Hospitals", "journal": "The Journal of Finance", "abstract": "<jats:title>ABSTRACT</jats:title><jats:p>The paper examines the decision\u2010making, compensation, and turnover of female CEOs in U.S. hospitals. Contrary to the literature on lower\u2010ranked executives and directors in public firms, there is no evidence that gender differences in preferences for risk or altruism affect decision\u2010making of hospital CEOs: corporate policies do not shift when women take (or leave) office, and male and female CEOs respond similarly to a major financial shock. However, female CEOs earn lower salaries, face flatter pay\u2010for\u2010performance incentives, and exhibit greater turnover after poor performance. Hospital boards behave as though they perceive female CEOs as less productive.</jats:p>", "volume": None, "issue": None, "authors": [["KATHARINA", "LEWELLEN", None]], "references": [{"reference_type": "article", "doi": "10.1016/j.jfineco.2008.10.007", "year": 2009, "title": "Women in the boardroom and their impact on governance and performance\u2606", "journal": "Journal of Financial Economics", "volume": "94", "issue": "2", "authors": [["Ren\u00e9e B.", "Adams", None], ["Daniel", "Ferreira", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1287/mnsc.1110.1452", "year": 2012, "title": "Beyond the Glass Ceiling: Does Gender Matter?", "journal": "Management Science", "volume": "58", "issue": "2", "authors": [["Ren\u00e9e B.", "Adams", "Australian School of Business, University of New South Wales, Sydney, NSW 2052, Australia"], ["Patricia", "Funk", "Universitat Pompeu Fabra and Barcelona Graduate School of Economics, 08005 Barcelona, Spain"]], "working_paper_institution": None}]}
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')

    @app.route('/load_example', methods=['GET'])
    def load_example():
        """Load example paper for demonstration."""
        global EXAMPLE_PAPER
        return jsonify(EXAMPLE_PAPER)

    @app.route('/recommend', methods=['POST'])
    def recommend():
        """Process paper information and generate citation recommendations."""
        global model, G, metadata, node_features, node_mapping, reverse_mapping, author_coauthor_G
        
        # Get paper information from form
        paper_data = request.json
        
        # Check required fields
        if not paper_data.get('title'):
            return jsonify({'error': 'Paper title is required'})
        
        if not paper_data.get('published_date'):
            return jsonify({'error': 'Publication date is required'})
        
        if not paper_data.get('authors') or len(paper_data.get('authors', [])) == 0:
            return jsonify({'error': 'At least one author is required'})
        
        # Create a temporary file to store the paper data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            json.dump(paper_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Get parameters from request
            mode = paper_data.get('mode', 'recommend')
            inductive_mode = paper_data.get('inductive', True)
            observed_ratio = float(paper_data.get('observed_ratio', 0.75))
            top_k = int(paper_data.get('top_k', 50))
            device = paper_data.get('device', 'cpu')
            
            # Process based on mode
            if mode == 'evaluate':
                if inductive_mode:
                    results = inductive_evaluate_with_held_out(
                        paper_data, 
                        G, 
                        metadata, 
                        model, 
                        node_features, 
                        node_mapping, 
                        reverse_mapping, 
                        author_coauthor_G,
                        observed_ratio=observed_ratio,
                        top_k=top_k,
                        device=device
                    )
                else:
                    results = evaluate_with_held_out(
                        paper_data, 
                        G, 
                        metadata, 
                        model, 
                        node_features, 
                        node_mapping, 
                        reverse_mapping, 
                        author_coauthor_G,
                        observed_ratio=observed_ratio,
                        top_k=top_k,
                        device=device
                    )
            else:
                if inductive_mode:
                    recommendations, paper_id, observed_citations, paper_metadata = inductive_recommend_citations(
                        paper_data, 
                        G, 
                        metadata, 
                        model, 
                        node_features, 
                        node_mapping, 
                        reverse_mapping, 
                        author_coauthor_G,
                        top_k=top_k,
                        device=device
                    )
                    
                    results = {
                        'paper_id': paper_id,
                        'paper_title': paper_metadata['title'],
                        'existing_citations': len(observed_citations),
                        'recommendations': recommendations
                    }
                else:
                    results = recommend_for_paper(
                        paper_data, 
                        G, 
                        metadata, 
                        model, 
                        node_features, 
                        node_mapping, 
                        reverse_mapping, 
                        author_coauthor_G,
                        top_k=top_k,
                        device=device
                    )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            return jsonify(results)
            
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
            return jsonify({'error': str(e)})
    
    return app

def load_full_example_paper():
    """Load the example paper directly from the JSONL file."""
    example_file_path = os.path.join(os.path.dirname(__file__), 'example_paper.jsonl')
    
    try:
        # Read the JSONL file - it should contain only one line
        with open(example_file_path, 'r', encoding='utf-8') as f:
            paper_json = f.readline().strip()
            return json.loads(paper_json)
    except Exception as e:
        print(f"Error loading example paper from file: {e}")
        # Return a fallback example paper if loading fails
        return {"doi": "10.1111/jofi.13455", "type": "journal-article", "published_date": "2025-04-01", "title": "Women in Charge: Evidence from Hospitals", "journal": "The Journal of Finance", "abstract": "<jats:title>ABSTRACT</jats:title><jats:p>The paper examines the decision\u2010making, compensation, and turnover of female CEOs in U.S. hospitals. Contrary to the literature on lower\u2010ranked executives and directors in public firms, there is no evidence that gender differences in preferences for risk or altruism affect decision\u2010making of hospital CEOs: corporate policies do not shift when women take (or leave) office, and male and female CEOs respond similarly to a major financial shock. However, female CEOs earn lower salaries, face flatter pay\u2010for\u2010performance incentives, and exhibit greater turnover after poor performance. Hospital boards behave as though they perceive female CEOs as less productive.</jats:p>", "volume": None, "issue": None, "authors": [["KATHARINA", "LEWELLEN", None]], "references": [{"reference_type": "article", "doi": "10.1016/j.jfineco.2008.10.007", "year": 2009, "title": "Women in the boardroom and their impact on governance and performance\u2606", "journal": "Journal of Financial Economics", "volume": "94", "issue": "2", "authors": [["Ren\u00e9e B.", "Adams", None], ["Daniel", "Ferreira", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1287/mnsc.1110.1452", "year": 2012, "title": "Beyond the Glass Ceiling: Does Gender Matter?", "journal": "Management Science", "volume": "58", "issue": "2", "authors": [["Ren\u00e9e B.", "Adams", "Australian School of Business, University of New South Wales, Sydney, NSW 2052, Australia"], ["Patricia", "Funk", "Universitat Pompeu Fabra and Barcelona Graduate School of Economics, 08005 Barcelona, Spain"]], "working_paper_institution": None}]}

def initialize_app():
    """Initialize the application with model and data."""
    global model, G, metadata, node_features, node_mapping, reverse_mapping, author_coauthor_G, EXAMPLE_PAPER
    
    try:
        # Load model and data
        model, G, metadata, node_features, node_mapping, reverse_mapping, author_coauthor_G = load_model_and_data()
        print("Model and data loaded successfully")
        
        # Load the full example paper
        EXAMPLE_PAPER = load_full_example_paper()
        print("Example paper loaded")
        
    except Exception as e:
        print(f"Error initializing model and data: {e}")

# Create the app
app = create_app()

# Initialize before first run
with app.app_context():
    initialize_app()

if __name__ == '__main__':
    app.run(debug=True) 
