from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db_1 = FAISS.load_local("literatureSearchAgentText", embeddings=embeddings, allow_dangerous_deserialization=True)
vector_db_2 = FAISS.load_local("summarizationAgentText", embeddings=embeddings, allow_dangerous_deserialization=True)
vector_db_3 = FAISS.load_local("hypothesisGenerationAgentText", embeddings=embeddings, allow_dangerous_deserialization=True)

stored_docs_1 = vector_db_1.docstore._dict
stored_docs_2 = vector_db_2.docstore._dict
stored_docs_3 = vector_db_3.docstore._dict
print("literature agent db")
for doc_id, doc in stored_docs_1.items():
    print(f"\n=== Document ID: {doc_id} ===")
    print(doc.page_content[:1000])
print('*'*50)
print("summarization agent db")
for doc_id, doc in stored_docs_1.items():
    print(f"\n=== Document ID: {doc_id} ===")
    print(doc.page_content[:1000])
print('*'*50)
print("hypothesis agent db")
for doc_id, doc in stored_docs_1.items():
    print(f"\n=== Document ID: {doc_id} ===")
    print(doc.page_content[:1000])