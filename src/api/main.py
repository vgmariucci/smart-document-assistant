# src/api/main.py
from fastapi import FastAPI
from src.models.rag_system import RAGSystem

app = FastAPI()
rag = RAGSystem.load_from_config()

@app.post("/query")
async def process_query(question: str):
    """Process natural language query"""
    results = rag.search(question)
    return {"question": question, "results": results}