from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_markdown_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def save_to_vector_db(text, path):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    docs = split_markdown_text(text)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(path)

def load_vector_db(path):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(path, embeddings=embeddings, allow_dangerous_deserialization=True)
    return vector_store