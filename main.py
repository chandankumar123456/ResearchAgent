#check the code and then run it first

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from Graph.GraphBuilder.graph_builder import create_research_workflow
from Graph.Nodes.state import ResearchState

def run_research_pipeline(query: str):
    """
    Run the complete research pipeline with the provided query.
    
    Args:
        query (str): The research topic or question to investigate
        
    Returns:
        str: The final compiled research report
    """
    # Create the workflow graph
    workflow = create_research_workflow()
    
    # Create the compiled workflow for execution
    app = workflow.compile()
    
    # Initialize the state with the query
    initial_state = ResearchState(
        query=query,
        literature_results="",
        summary_results="",
        hypothesis_results="",
        errors=[],
        final_output=""
    )
    
    # Execute the workflow
    try:
        print(f"Starting research pipeline for query: '{query}'")
        result = app.invoke(initial_state)
        
        # Check if there were errors
        if result.errors:
            print("Research pipeline completed with errors:")
            for error in result.errors:
                print(f"- {error}")
            return result.final_output
        
        print("Research pipeline completed successfully!")
        return result.final_output
    
    except Exception as e:
        print(f"Error executing research pipeline: {str(e)}")
        return f"Research pipeline failed: {str(e)}"

if __name__ == "__main__":
    print("======= AgenticAI Research Assistant =======\n")
    
    # Get research query from user
    query = input("Enter your research topic or question: ")
    
    # Run the research pipeline
    result = run_research_pipeline(query)
    
    # Display and save results
    print("\n======= Research Report =======\n")
    print(result)
    
    # # Save the report to a file
    # try:
    #     with open("research_report.md", "w") as f:
    #         f.write(result)
    #     print("\nResearch report saved to 'research_report.md'")
    # except Exception as e:
    #     print(f"\nError saving report to file: {str(e)}")