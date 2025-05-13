import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from state import ResearchState

from Agents.hypothesisGenerationAgent import generate_hypothesis_from_summaries

def hypothesis_generation_node(state: ResearchState) -> ResearchState:
    """Run the hypothesis generation agent and update the state."""
    try:
        hypothesis_results = generate_hypothesis_from_summaries(state.query)
        state.hypothesis_results = hypothesis_results
        return state
    except Exception as e:
        return {"errors": state.errors + [f"Hypothesis Generation error: {str(e)}"]}