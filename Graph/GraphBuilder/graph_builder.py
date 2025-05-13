import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from langgraph.graph import StateGraph, START, END

from Graph.Nodes.state import ResearchState
from Graph.Nodes.compile_results_node import compile_results_node
from Graph.Nodes.hypothesis_generation_node import hypothesis_generation_node
from Graph.Nodes.literature_search_node import literature_search_node
from Graph.Nodes.summarization_node import summarization_node

from utils.loadapi import loadapi
loadapi()

def should_continue(state: ResearchState) -> str:
    """Decide if we should continue or end due to errors."""
    if state.errors:
        return "handle_errors"
    return "continue"

def handle_errors_node(state: ResearchState) -> ResearchState:
    """Create an error report."""
    error_report = "## Errors Encountered\n\n"
    for error in state.errors:
        error_report += f"- {error}\n"
    
    state.final_output = error_report
    return state

def create_research_workflow():
    """Create and return a LangGraph workflow for the research Process."""
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("literature_search", literature_search_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("hypothesis_generation", hypothesis_generation_node)
    graph.add_node("compile_results", compile_results_node)
    graph.add_node("handle_errors", handle_errors_node)
    
    # Add edges
    graph.add_edge(START, "literature_search")
    graph.add_edge("literature_search", "summarization")
    graph.add_edge("summarization", "hypothesis_generation")
    graph.add_edge("hypothesis_generation", "compile_results")
    graph.add_edge("compile_results", END)
    graph.add_edge("handle_errors", END)
    
    # Add conditionanl Edges
    graph.add_conditional_edges(
        "literature_search",
        should_continue,
        {
            "continue": "summarization",
            "handle_errors": "handle_errors"
        }
    )
    
    graph.add_conditional_edges(
        "summarization",
        should_continue,
        {
            "continue": "hypothesis_generation",
            "handle_errors": "handle_errors"
        }
    )
    
    graph.add_conditional_edges(
        "hypothesis_generation",
        should_continue,
        {
            "continue": "compile_results",
            "handle_errors": "handle_errors"
        }
    )
    
    return graph

graph = create_research_workflow()