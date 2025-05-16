#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for citation recommendation using evaluate_gnn.py.

This script demonstrates how to use the citation recommendation system
with the example paper from the original task.
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
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Example paper from the task
    example_paper = {"doi": "10.1016/j.jfineco.2024.103897", "type": "journal-article", "published_date": "2024-09-01", "title": "Are cryptos different? Evidence from retail trading", "journal": "Journal of Financial Economics", "abstract": None, "volume": "159", "issue": None, "authors": [["Shimon", "Kogan", None], ["Igor", "Makarov", None], ["Marina", "Niessner", None], ["Antoinette", "Schoar", None]], "references": [{"reference_type": "book", "doi": "10.1016/b978-0-12-822927-9.00024-0", "year": 2023, "title": "Expectations data in asset pricing", "journal": "Handbook of Economic Expectations", "volume": None, "issue": None, "authors": [["Klaus", "Adam", None], ["Stefan", "Nagel", None]], "working_paper_institution": None}, {"reference_type": "working_paper", "doi": "10.3386/w31856", "year": None, "title": "Who Invests in Crypto? Wealth, Financial Constraints, and Risk Attitudes", "journal": None, "volume": None, "issue": None, "authors": [["Darren", "Aiello", None], ["Scott", "Baker", None], ["Tetyana", "Balyuk", None], ["Marco Di", "Maggio", None], ["Mark", "Johnson", None], ["Jason", "Kotter", None]], "working_paper_institution": None}, {"title": "Regulating cryptocurrencies: Assessing market reactions", "author": "Auer", "year": "2018", "journal": "BIS Q. Rev. Sept.", "reference_type": "article"}, {"reference_type": "article", "doi": "10.1111/0022-1082.00226", "year": 2000, "title": "Trading Is Hazardous to Your Wealth: The Common Stock Investment Performance of Individual Investors", "journal": "The Journal of Finance", "volume": "55", "issue": "2", "authors": [["Brad M.", "Barber", None], ["Terrance", "Odean", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/j.jfineco.2018.04.007", "year": 2018, "title": "Extrapolation and bubbles", "journal": "Journal of Financial Economics", "volume": "129", "issue": "2", "authors": [["Nicholas", "Barberis", None], ["Robin", "Greenwood", None], ["Lawrence", "Jin", None], ["Andrei", "Shleifer", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/s0304-405x(98)00027-0", "year": 1998, "title": "A model of investor sentiment1We are grateful to the NSF for financial support, and to Oliver Blanchard, Alon Brav, John Campbell (a referee), John Cochrane, Edward Glaeser, J.B. Heaton, Danny Kahneman, David Laibson, Owen Lamont, Drazen Prelec, Jay Ritter (a referee), Ken Singleton, Dick Thaler, an anonymous referee, and the editor, Bill Schwert, for comments.1", "journal": "Journal of Financial Economics", "volume": "49", "issue": "3", "authors": [["Nicholas", "Barberis", None], ["Andrei", "Shleifer", None], ["Robert", "Vishny", None]], "working_paper_institution": None}, {"reference_type": "book", "doi": "10.1016/s1574-0102(03)01027-6", "year": 2003, "title": "Chapter 18 A survey of behavioral finance", "journal": "Handbook of the Economics of Finance", "volume": None, "issue": None, "authors": [["Nicholas", "Barberis", None], ["Richard", "Thaler", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/j.1540-6261.2009.01448.x", "year": 2009, "title": "What Drives the Disposition Effect? An Analysis of a Long\u2010Standing Preference\u2010Based Explanation", "journal": "The Journal of Finance", "volume": "64", "issue": "2", "authors": [["NICHOLAS", "BARBERIS", None], ["WEI", "XIONG", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/j.jfineco.2021.07.013", "year": 2022, "title": "Retail shareholder participation in the proxy process: Monitoring, engagement, and voting", "journal": "Journal of Financial Economics", "volume": "144", "issue": "2", "authors": [["Alon", "Brav", None], ["Matthew", "Cain", None], ["Jonathon", "Zytnick", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1257/aer.99.2.393", "year": 2009, "title": "Measuring the Financial Sophistication of Households", "journal": "American Economic Review", "volume": "99", "issue": "2", "authors": [["Laurent E", "Calvet", "Department of Finance, HEC Paris, 1 avenue de la Lib\u00e9ration, 78351 Jouy-en-Josas, France, and NBER."], ["John Y", "Campbell", "Department of Economics, Harvard University, Littauer Center, Cambridge, MA 02138, and NBER."], ["Paolo", "Sodini", "Department of Finance, Stockholm School of Economics, Sveav\u00e4gen 65, Box 6501, SE-113 83 Stockholm, Sweden."]], "working_paper_institution": None}, {"reference_type": "book", "doi": "10.1016/b978-012174597-4/50030-2", "year": 2002, "title": "Author Index", "journal": "Principles of Cloning", "volume": None, "issue": None, "authors": [["J", "CIBELLI", None], ["R", "LANZA", None], ["K", "CAMPBELL", None], ["M", "WEST", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1093/rfs/hhaa089", "year": 2021, "title": "Tokenomics: Dynamic Adoption and Valuation", "journal": "The Review of Financial Studies", "volume": "34", "issue": "3", "authors": [["Lin William", "Cong", "SC Johnson College of Business, Cornell University"], ["Ye", "Li", "Fisher College of Business, The Ohio State University"], ["Neng", "Wang", "Columbia Business School and NBER"]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/j.jfineco.2024.103870", "year": 2024, "title": "The social signal", "journal": "Journal of Financial Economics", "volume": "158", "issue": None, "authors": [["J. Anthony", "Cookson", None], ["Runjing", "Lu", None], ["William", "Mullins", None], ["Marina", "Niessner", None]], "working_paper_institution": None}, {"reference_type": "book", "doi": "10.1016/b978-0-444-50897-3.50009-2", "year": 2010, "title": "Heterogeneity and Portfolio Choice: Theory and Evidence", "journal": "Handbook of Financial Econometrics: Tools and Techniques", "volume": None, "issue": None, "authors": [["Stephanie", "Curcuru", None], ["John", "Heaton", None], ["Deborah", "Lucas", None], ["Damien", "Moore", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/j.jfineco.2020.10.003", "year": 2021, "title": "Extrapolative beliefs in the cross-section: What can we learn from the crowds?", "journal": "Journal of Financial Economics", "volume": "140", "issue": "1", "authors": [["Zhi", "Da", None], ["Xing", "Huang", None], ["Lawrence J.", "Jin", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/0022-1082.00077", "year": 1998, "title": "Investor Psychology and Security Market Under\u2010 and Overreactions", "journal": "The Journal of Finance", "volume": "53", "issue": "6", "authors": [["Kent", "Daniel", None], ["David", "Hirshleifer", None], ["Avanidhar", "Subrahmanyam", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1002/jae.1225", "year": 2011, "title": "Measuring and interpreting expectations of equity returns", "journal": "Journal of Applied Econometrics", "volume": "26", "issue": "3", "authors": [["Jeff", "Dominitz", None], ["Charles F.", "Manski", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1287/mnsc.2014.1979", "year": 2015, "title": "Trading as Gambling", "journal": "Management Science", "volume": "61", "issue": "10", "authors": [["Anne Jones", "Dorn", "Philadelphia, Pennsylvania 19104"], ["Daniel", "Dorn", "LeBow College of Business, Drexel University, Philadelphia, Pennsylvania 19104"], ["Paul", "Sengmueller", "CentER, Tilburg University, 5037 AB Tilburg, The Netherlands"]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1257/mic.2.4.221", "year": 2010, "title": "Na\u00efve Herding in Rich-Information Settings", "journal": "American Economic Journal: Microeconomics", "volume": "2", "issue": "4", "authors": [["Erik", "Eyster", "Department of Economics, London School of Economics, Houghton Street, London WC2A 2AE United Kingdom."], ["Matthew", "Rabin", "Department of Economics, University of California, Berkeley, 549 Evans Hall, #3880 Berkeley, CA 94720-3880."]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1086/663989", "year": 2012, "title": "Natural Expectations, Macroeconomic Dynamics, and Asset Pricing", "journal": "NBER Macroeconomics Annual", "volume": "26", "issue": "1", "authors": [["Andreas", "Fuster", None], ["Benjamin", "Hebert", None], ["David", "Laibson", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1257/aer.20200243", "year": 2021, "title": "Five Facts about Beliefs and Portfolios", "journal": "American Economic Review", "volume": "111", "issue": "5", "authors": [["Stefano", "Giglio", "Yale School of Management, NBER, CEPR (email: )"], ["Matteo", "Maggiori", "Graduate School of Business, Stanford University, NBER, CEPR (email: )"], ["Johannes", "Stroebel", "Stern School of Business, New York University, NBER, CEPR (email: )"], ["Stephen", "Utkus", "Vanguard (email: )"]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1093/qje/qju035", "year": 2015, "title": "Waves in Ship Prices and Investment\n*", "journal": "The Quarterly Journal of Economics", "volume": "130", "issue": "1", "authors": [["Robin", "Greenwood", None], ["Samuel G.", "Hanson", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1093/rfs/hht082", "year": 2014, "title": "Expectations of Returns and Expected Returns", "journal": "Review of Financial Studies", "volume": "27", "issue": "3", "authors": [["Robin", "Greenwood", None], ["Andrei", "Shleifer", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/jofi.12903", "year": 2020, "title": "Is Bitcoin Really Untethered?", "journal": "The Journal of Finance", "volume": "75", "issue": "4", "authors": [["JOHN M.", "GRIFFIN", None], ["AMIN", "SHAMS", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/s0304-405x(99)00044-6", "year": 2000, "title": "The investment behavior and performance of various investor types: a study of Finland's unique data set", "journal": "Journal of Financial Economics", "volume": "55", "issue": "1", "authors": [["M", "Grinblatt", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1093/rof/rfab034", "year": 2022, "title": "The Characteristics and Portfolio Behavior of Bitcoin Investors: Evidence from Indirect Cryptocurrency Investments", "journal": "Review of Finance", "volume": "26", "issue": "4", "authors": [["Andreas", "Hackethal", "Goethe University Frankfurt , Frankfurt, Germany"], ["Tobin", "Hanspal", "WU Vienna University of Economics and Business , Vienna, Austria"], ["Dominique M", "Lammer", "Goethe University Frankfurt , Frankfurt, Germany"], ["Kevin", "Rink", "Goethe University Frankfurt , Frankfurt, Germany"]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1017/s0022109021000077", "year": 2022, "title": "Social Transmission Bias and Investor Behavior", "journal": "Journal of Financial and Quantitative Analysis", "volume": "57", "issue": "1", "authors": [["Bing", "Han", None], ["David", "Hirshleifer", None], ["Johan", "Walden", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1093/rfs/5.3.473", "year": 1993, "title": "Differences of Opinion Make a Horse Race", "journal": "Review of Financial Studies", "volume": "6", "issue": "3", "authors": [["Milton", "Harris", None], ["Artur", "Raviv", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1093/rfs/hhu079", "year": 2015, "title": "The Worst, the Best, Ignoring All the Rest: The Rank Effect and Trading Behavior", "journal": "The Review of Financial Studies", "volume": "28", "issue": "4", "authors": [["Samuel M.", "Hartzmark", "University of Chicago Booth School of Business"]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/j.1540-6261.2006.00867.x", "year": 2006, "title": "Asset Float and Speculative Bubbles", "journal": "The Journal of Finance", "volume": "61", "issue": "3", "authors": [["HARRISON", "HONG", None], ["JOS\u00c9", "SCHEINKMAN", None], ["WEI", "XIONG", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/j.1540-6261.2008.01316.x", "year": 2008, "title": "Individual Investor Trading and Stock Returns", "journal": "The Journal of Finance", "volume": "63", "issue": "1", "authors": [["RON", "KANIEL", None], ["GIDEON", "SAAR", None], ["SHERIDAN", "TITMAN", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/j.1540-6261.2009.01483.x", "year": 2009, "title": "Who Gambles in the Stock Market?", "journal": "The Journal of Finance", "volume": "64", "issue": "4", "authors": [["ALOK", "KUMAR", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1093/rfs/hhaa113", "year": 2021, "title": "Risks and Returns of Cryptocurrency", "journal": "The Review of Financial Studies", "volume": "34", "issue": "6", "authors": [["Yukun", "Liu", "University of Rochester"], ["Aleh", "Tsyvinski", "Yale University"]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/j.jfineco.2019.07.001", "year": 2020, "title": "Trading and arbitrage in cryptocurrency markets", "journal": "Journal of Financial Economics", "volume": "135", "issue": "2", "authors": [["Igor", "Makarov", None], ["Antoinette", "Schoar", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/jofi.13179", "year": 2022, "title": "Belief Disagreement and Portfolio Choice", "journal": "The Journal of Finance", "volume": "77", "issue": "6", "authors": [["MAARTEN", "MEEUWIS", None], ["JONATHAN A.", "PARKER", None], ["ANTOINETTE", "SCHOAR", None], ["DUNCAN", "SIMESTER", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1093/revfin/hhm011", "year": 2007, "title": "Equilibrium Underdiversification and the Preference for Skewness", "journal": "Review of Financial Studies", "volume": "20", "issue": "4", "authors": [["Todd", "Mitton", None], ["Keith", "Vorkink", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/0022-1082.00072", "year": 1998, "title": "Are Investors Reluctant to Realize Their Losses?", "journal": "The Journal of Finance", "volume": "53", "issue": "5", "authors": [["Terrance", "Odean", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1016/j.jfineco.2005.05.003", "year": 2006, "title": "Investor attention, overconfidence and category learning", "journal": "Journal of Financial Economics", "volume": "80", "issue": "3", "authors": [["Lin", "Peng", None], ["Wei", "Xiong", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1086/378531", "year": 2003, "title": "Overconfidence and Speculative Bubbles", "journal": "Journal of Political Economy", "volume": "111", "issue": "6", "authors": [["Jos\u00e9\u00a0A.", "Scheinkman", None], ["Wei", "Xiong", None]], "working_paper_institution": None}, {"reference_type": "article", "doi": "10.1111/jofi.13192", "year": 2023, "title": "Decentralization through Tokenization", "journal": "The Journal of Finance", "volume": "78", "issue": "1", "authors": [["MICHAEL", "SOCKIN", None], ["WEI", "XIONG", None]], "working_paper_institution": None}], "reference_stats": {"total_references": 55, "parsed_references": 40, "skipped_references": 15}}
    
    # Save example paper to a file
    example_paper_path = os.path.join(args.output_dir, 'example_paper.json')
    with open(example_paper_path, 'w') as f:
        json.dump(example_paper, f, indent=2)
    
    print(f"Saved example paper to {example_paper_path}")
    
    # Run evaluate_gnn.py in evaluation mode
    evaluation_output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    cmd = f"python evaluate_gnn.py --input {example_paper_path} --output {evaluation_output_path} --mode evaluate --observed-ratio {args.observed_ratio} --top-k {args.top_k} --device {args.device}"
    
    print(f"Running evaluation: {cmd}")
    os.system(cmd)
    
    # Process the evaluation results to mark citations
    with open(evaluation_output_path, 'r') as f:
        evaluation_results = json.load(f)
    
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
    cmd = f"python evaluate_gnn.py --input {example_paper_path} --output {recommendation_output_path} --mode recommend --top-k {args.top_k} --device {args.device}"
    
    print(f"Running recommendation: {cmd}")
    os.system(cmd)
    
    # Process the recommendation results
    with open(recommendation_output_path, 'r') as f:
        recommendation_results = json.load(f)
    
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
    print(f"Total held-out citations: {len(held_out_citations)}")
    actual_citations_found = sum(1 for rec in evaluation_results.get('recommendations', []) if rec.get('status') == 'ACTUAL_CITATION')
    print(f"Actual citations found in top {args.top_k}: {actual_citations_found}")
    
    # Print top recommendations with status markers
    print("\nTop Evaluation Recommendations:")
    print("-" * 80)
    for rec in evaluation_results.get('recommendations', [])[:10]:  # Show top 10 for brevity
        status_marker = rec.get('checkbox', '')
        print(f"{rec.get('rank', 0)}. {status_marker} {rec.get('title', 'Unknown Title')} (Score: {rec.get('score', 0):.4f})")
        print(f"   Authors: {rec.get('authors', '')}")
        print(f"   Journal: {rec.get('journal', '')}")
        print(f"   Status: {rec.get('status', '')}")
        print()
    
    print("\nRecommendation Results Summary:")
    already_cited = sum(1 for rec in recommendation_results.get('recommendations', []) if rec.get('status') == 'ALREADY_CITED')
    new_recommendations = sum(1 for rec in recommendation_results.get('recommendations', []) if rec.get('status') == 'NEW_RECOMMENDATION')
    print(f"Already cited papers in recommendations: {already_cited}")
    print(f"New recommendations: {new_recommendations}")
    
    # Print top recommendations with status markers
    print("\nTop Recommendation Results:")
    print("-" * 80)
    for rec in recommendation_results.get('recommendations', [])[:10]:  # Show top 10 for brevity
        status_marker = rec.get('checkbox', '')
        print(f"{rec.get('rank', 0)}. {status_marker} {rec.get('title', 'Unknown Title')} (Score: {rec.get('score', 0):.4f})")
        print(f"   Authors: {rec.get('authors', '')}")
        print(f"   Journal: {rec.get('journal', '')}")
        print(f"   Status: {rec.get('status', '')}")
        print()
    
    print("\nTest complete!")
    print(f"Evaluation results saved to {evaluation_output_path}")
    print(f"Recommendation results saved to {recommendation_output_path}")
    print(f"Legend: [✓]=Actual citation that was held out, [ ]=Not cited, [C]=Already cited, [N]=New recommendation")

if __name__ == "__main__":
    main() 
