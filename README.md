# Intelligent Document Analysis System

This project demonstrates a comprehensive document analysis system combining classical NLP techniques with modern LLM capabilities.

## Features

- **Document Processing Pipeline**: Cleans, preprocesses, and vectorizes text documents
- **Classical ML Models**: Example document classification using Random Forest
- **LLM Integration**: RAG (Retrieval-Augmented Generation) system using LangChain
- **Production API**: FastAPI endpoint for querying the system
- **Testing**: Unit tests and integration tests

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your documents in `data/raw/` (sample documents provided)
4. Run the processing pipeline: `python src/data_processing/document_processor.py`
5. Initialize the RAG system: `python src/models/rag_system.py`
6. Start the API: `python src/api/main.py`

## Usage

- Access the API at `http://localhost:8000/docs` for interactive documentation
- Query the system with natural language questions
- Explore the Jupyter notebooks for additional examples

## Technologies Used

- Python
- spaCy (NLP)
- scikit-learn (ML)
- LangChain (LLM orchestration)
- FAISS (Vector storage)
- FastAPI (Production API)
- pytest (Testing)