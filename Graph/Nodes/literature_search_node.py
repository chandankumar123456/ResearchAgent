import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from state import ResearchState

from Agents.literatureSearchAgent import run_literature_search

def literature_search_node(state: ResearchState) -> ResearchState:
    """Run the literature search agent and update the state."""
    try:
        literature_results = run_literature_search(query=state.query)
        state.literature_results = literature_results
        return state
    except Exception as e:
        return {"errors": state.errors + [f"Literature Search Error: {str(e)}"]}
    