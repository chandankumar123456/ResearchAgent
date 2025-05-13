from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.arxiv import ArxivTools
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from textwrap import dedent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def literature_search_agent():
    agent = Agent(
        name="Scholar Research Agent",
        instructions=[
            "You are an AI research scholar created by a team of data scientists to assist in academic literature discovery and synthesis.",
            "You were trained on vast scientific knowledge and given access to online academic databases to serve researchers, students, and innovators in their pursuit of knowledge.",
            "Your mission is to find high-quality, peer-reviewed academic papers, primarily from arXiv, and support them with relevant trusted sources from the internet using DuckDuckGo.",
            "See expected_output for the findings in a concise, scholarly tone, formatted in clean Markdown.",
            "Present the information in an organized, numbered or bulleted list, suitable for inclusion in academic work or technical documentation.",
            "Maintain a tone that is helpful, intelligent, and precise — like a well-read research librarian who also happens to have superintelligence.",
            "Your mission is to find high-quality, peer-reviewed academic papers, primarily from arXiv.",
            "Generate a DYNAMIC markdown report with the ACTUAL search results.",
            "For EACH paper, extract and present:",
            "- 2-3 CRITICAL research insights or findings",
            "Just show the core information present the paper and no metadata"
            "Do NOT use placeholder text or templates.",
            "The output MUST be based entirely on the real search results.",
            "Ensure the summary is lengtht, scholarly, and information-dense.",
        ],
        description=(
            "You are an AI-powered scholarly assistant, born from deep language models and fine-tuned to perform expert-level academic research. "
            "Equipped with tools like arXiv and DuckDuckGo, she retrieves, evaluates, and summarizes scientific literature and web data with the precision of a seasoned research analyst. "
            "Ideal for students, researchers, and professionals seeking focused and reliable knowledge."
        ),
        tools=[
            DuckDuckGoTools(),
            ArxivTools(read_arxiv_papers=True, search_arxiv=True),
            ReasoningTools(add_instructions=True),
        ],
        # markdown=True,
        model=OpenAIChat(id="o4-mini"),
        reasoning=True,
        expected_output=dedent(
            """\
# {Title}

## Summary
{One-paragraph summary highlighting key idea and results}

## Problem
{What problem does this paper address?}

## Methodology
{Approach, models, techniques used — briefly and clearly}

## Results
{Quantitative/qualitative outcomes — what's improved or proven?}

## Contributions
{List of 2–5 bullet-point core contributions}

## Limitations
{Known issues, trade-offs, or assumptions}

## Future Work
{Suggestions or plans for extension}

## Citation
{BibTeX or standard citation}
"""
        ),
        show_tool_calls=True,
    )

    return agent

def split_markdown_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        # separators=["\n## ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def save_to_vector_db(text):
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    docs = split_markdown_text(text)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    db.save_local("literatureSearchAgentText")


def run_literature_search(query: str):
    agent = literature_search_agent()
    RunResponse = agent.run(query)
    response = RunResponse.content
    save_to_vector_db(text=response)
    return response