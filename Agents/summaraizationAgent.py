from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.arxiv import ArxivTools

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate


def load_vector_store(path: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
    return vector_store

vector_store = load_vector_store("../literatureSearchAgentText")
retriever = vector_store.as_retriever()
query = "how vision transformers are different from cnn's and object detections"
documents = retriever.invoke(query)

combined_text = "\n\n".join(doc.page_content for doc in documents)

llm = ChatOpenAI(model="o4-mini", temperature=1.0)
prompt = PromptTemplate.from_template("""
This is the user query: {query}
This is the context: {combined_text}

If the context contains relevant information to answer the query, then answer it.
If not, say 'i dont know'.
""")
chain = prompt | llm | StrOutputParser()

response = chain.invoke({
    "query": query,
    "combined_text": combined_text
})
print(response)