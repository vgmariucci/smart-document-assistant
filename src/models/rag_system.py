# src/models/rag_system.py
import os
import pandas as pd 
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from src.utils.logger import setup_logger
from src.utils.config import MODEL_DIR

logger = setup_logger("rag_system")

class DocumentRAG:
    """
    Retrieval-Augmented Generation (RAG) system for intelligent document analysis
    """
    
    def __init__(self, openai_api_key=None):
        """Initialize the RAG system"""
        logger.info("Initializing RAG system")
        
        # Set API key from environment or parameter
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Some functionality will be limited.")
        
        # Initialize components
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.qa_chain = None
    
    def _prepare_documents(self, df):
        """Prepare documents from DataFrame for embedding"""
        logger.info("Preparing documents from DataFrame")
        
        # Convert DataFrame to documents
        loader = DataFrameLoader(df, page_content_column="document_text")
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(split_documents)} document chunks")
        
        return split_documents
    
    def build_index(self, df, document_text_column=None):
        """Build a vector index from the DataFrame"""
        logger.info("Building vector index")
        
        # If a specific text column is provided, use it as the document text
        if document_text_column:
            # Create a document text column by combining relevant information
            df = df.copy()
            df['document_text'] = df[document_text_column].astype(str)
        else:
            # Create a document text column by combining relevant information
            df = df.copy()
            
            # Combine various columns to create a rich text representation
            df['document_text'] = df.apply(
                lambda row: (
                    f"Transaction ID: {row['transaction_id']}\n"
                    f"Date: {row['transaction_date']}\n"
                    f"Product: {row['product_category']} - {row['product_brand']}\n"
                    f"Price: ${row['unit_price']} x {row['quantity']} = ${row['total_price']}\n"
                    f"Customer: {row['customer_id']} ({row['customer_type']})\n"
                    f"Payment: {row['payment_method']}\n"
                    f"Location: {row['store_location']}\n"
                ),
                axis=1
            )
        
        # Prepare documents
        documents = self._prepare_documents(df)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        
        # Initialize retrieval QA chain if API key is available
        if self.openai_api_key:
            llm = OpenAI(openai_api_key=self.openai_api_key)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever()
            )
        
        logger.info("Vector index built successfully")
        return True
    
    def save_index(self, index_name="document_index"):
        """Save the vector index to disk"""
        if self.vector_store is None:
            logger.error("No index to save")
            return False
        
        save_path = os.path.join(MODEL_DIR, index_name)
        logger.info(f"Saving index to {save_path}")
        
        try:
            self.vector_store.save_local(save_path)
            logger.info("Index saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, index_name="document_index"):
        """Load a vector index from disk"""
        load_path = os.path.join(MODEL_DIR, index_name)
        logger.info(f"Loading index from {load_path}")
        
        try:
            self.vector_store = FAISS.load_local(load_path, self.embedding_model)
            
            # Initialize retrieval QA chain if API key is available
            if self.openai_api_key:
                llm = OpenAI(openai_api_key=self.openai_api_key)
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever()
                )
            
            logger.info("Index loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def search_documents(self, query, k=5):
        """Search for similar documents"""
        if self.vector_store is None:
            logger.error("No vector store available for search")
            return []
        
        logger.info(f"Searching for documents with query: '{query}'")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def ask(self, question):
        """Ask a question using the QA chain"""
        if self.qa_chain is None:
            if self.openai_api_key is None:
                logger.error("No OpenAI API key provided for QA functionality")
                return "Error: OpenAI API key is required for QA functionality"
            elif self.vector_store is None:
                logger.error("No vector store available for QA")
                return "Error: No document index loaded for answering questions"
        
        logger.info(f"Asking question: '{question}'")
        
        try:
            result = self.qa_chain({"query": question})
            return result['result']
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return f"Error: {str(e)}"