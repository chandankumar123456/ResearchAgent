import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from state import ResearchState

from Agents.summarizationAgent import summarize_search_docs
    
def summarization_node(state: ResearchState) -> ResearchState:
    """Run the summarization agent and update the state."""
    try:
        summary_results = summarize_search_docs(state.query)
        state.summary_results = summary_results
        return state
    except Exception as e:
        return {"errors": state.errors + [f"Summarization error: {str(e)}"]}