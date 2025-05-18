# src/models/rag_system.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class RAGSystem:
    def __init__(self, config):
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_db = FAISS

    def create_vector_store(self, documents):
        """Create FAISS vector store from documents"""
        return self.vector_db.from_documents(
            documents=documents,
            embedding=self.embeddings
        )