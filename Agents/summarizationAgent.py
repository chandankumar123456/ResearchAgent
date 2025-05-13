import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from literatureSearchAgent import run_literature_search
from utils.vector_db_helper import save_to_vector_db
from utils.vector_db_helper import load_vector_db

def summarize_search_docs(query: str):
    literature_response = run_literature_search(query=query)
    vector_store = load_vector_db("literatureSearchAgentText")
    retriever = vector_store.as_retriever()
    documents = retriever.invoke(query)
    
    combined_docs = "\n\n".join(doc.page_content for doc in documents)
    llm = ChatOpenAI(model="o4-mini", temperature=1)
    prompt = PromptTemplate.from_template("""
        This is the user query: {query}
        This is the context: {combined_docs}

        If the context contains relevant information to answer the query, then answer it.
        If not, say 'i dont know'.                  
    """)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "query": query,
        "combined_docs": combined_docs
    })
    save_to_vector_db(response, "summarizationAgentText")
    return response

query: str = input("Enter a query: ")
print(summarize_search_docs(query=query))