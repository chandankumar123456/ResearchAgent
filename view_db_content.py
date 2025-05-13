from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = FAISS.load_local("Agents/literatureSearchAgentText", embeddings=embeddings, allow_dangerous_deserialization=True)

stored_docs = vector_db.docstore._dict
for doc_id, doc in stored_docs.items():
    print(f"\n=== Document ID: {doc_id} ===")
    print(doc.page_content[:1000])