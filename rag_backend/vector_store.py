from langchain_community.vectorstores import Chroma
from embedding import dense_embeddings
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_store_path = os.path.join(base_dir, "compliance_db")

def vector_store(new_docs):

    if os.path.exists(vector_store_path):
        print("Vector store is already existing. Adding the document to existing vector store....")

        vector_store = Chroma(
            persist_directory = vector_store_path,
            embedding_function = dense_embeddings
        )

        vector_store.add_documents(new_docs)

    else:
        print("vector store is not existed. Creating new vector store....")

        vector_store = Chroma.from_documents(
            documents = new_docs,
            embedding = dense_embeddings,
            persist_directory = "compliance_db"
        )

    print("Vector Store has been created/updated and added the documents successfully.")