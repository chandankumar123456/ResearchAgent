import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from state import ResearchState

def compile_results_node(state: ResearchState) -> ResearchState:
    """Compile all results into a final output."""
    final_output = f"""# Research Analysis for: {state.query}

    ## Literature Search Results

    {state.literature_results}

    ## Summary of Research

    {state.summary_results}

    ## Research Hypotheses and Gaps

    {state.hypothesis_results}
"""
    return state