import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from textwrap import dedent

from utils.vector_db_helper import load_vector_db, save_to_vector_db
from summarizationAgent import summarize_search_docs
def hypothesis_generation_agent():
    agent = Agent(
        name="Research Hypothesis Generator",
        instructions=[
            "You are an advanced Research Hypothesis Generator that analyzes academic literature and identifies research gaps.",
            "Your primary role is to identify promising future research directions based on literature summaries.",
            "Using the provided literature summaries, you will:",
            "1. Identify methodological gaps in existing research",
            "2. Detect under-explored application domains",
            "3. Recognize limitations in current approaches",
            "4. Propose novel combinations of existing methods",
            "5. Generate testable hypotheses for future research",
            "Your output should be comprehensive, scholarly, and reflect deep understanding of the research domain.",
            "Format your response in clean Markdown with clear sections.",
            "Maintain an objective, scholarly tone throughout your analysis."
        ],
        description=(
            "I analyze academic literature summaries to identify research gaps and generate "
            "promising hypotheses for future investigation. By detecting methodological limitations, "
            "under-explored domains, and potential novel approaches, I help researchers discover "
            "new research directions and opportunities for innovation."
        ),
        tools=[ReasoningTools(add_instructions=True)],
        model=OpenAIChat(id="o4-mini"),
        reasoning=True,
        expected_output=dedent(
            """\
            # Research Gap Analysis and Hypothesis Generation

            ## Identified Research Gaps
            {List 3-5 specific research gaps with brief explanations}

            ## Methodological Limitations
            {List 2-3 limitations in current methodological approaches}

            ## Underexplored Applications
            {List 2-3 promising application domains that need more attention}

            ## Proposed Hypotheses
            {Generate 3-5 specific, testable research hypotheses}

            ## Future Research Directions
            {Suggest 2-3 promising research directions with rationale}

            ## Integration Opportunities
            {Identify 2-3 opportunities for integrating different approaches}
            """
        ),
        show_tool_calls=True,
    )
    
    return agent

def generate_hypothesis_from_summaries(query: str):
    """
    Generate research hypotheses based on the summarization results stored in the vector database.
    """
    try:
        # summarize_agent_response = summarize_search_docs(query=query)
        vector_store = load_vector_db("summarizationAgentText")
    except Exception as e:
        print(f"Error loading summarization vector database: {e}")
        return "Could not load summarization results. Please run the summarization agent first."

    # Retrieve Relevant Summaries
    retriever = vector_store.as_retriever()
    documents = retriever.invoke(query)
    
    if not documents:
        return "No relevant summaries found in the database. Please run the summarization agent with a valid query first."

    # Combine the Documents
    combined_docs = "\n\n".join(doc.page_content for doc in documents)

    # Use the agent to generate hypotheses
    agent = hypothesis_generation_agent()
    enhanced_query = (
        f"Based on the following research summaries, identify research gaps and generate "
        f"hypotheses for future research on the topic: '{query}'\n\n"
        f"Research Summaries:\n{combined_docs}"
    )

    try:
        run_response = agent.run(enhanced_query)
        response = run_response.content
        
        # Save the generated hypotheses to a vector database for later use
        save_to_vector_db(text=response, path="hypothesisGenerationAgentText")
        
        return response
    except Exception as e:
        print(f"Error in hypothesis generation: {e}")
        return f"Error generating hypotheses: {str(e)}"


if __name__ == "__main__":
    query = input("Enter your research query: ")
    hypotheses = generate_hypothesis_from_summaries(query)
    print("\n" + "="*50 + "\n")
    print(hypotheses)
