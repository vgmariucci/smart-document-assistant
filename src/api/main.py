# src/api/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
import pandas as pd
import json
from src.data_processing.data_loader import DataLoader
from src.data_processing.document_processor import DocumentProcessor
from src.models.classical_ml import SalesPredictor
from src.models.rag_system import DocumentRAG
from src.utils.logger import setup_logger
from src.utils.config import API_HOST, API_PORT
from .schemas import (
    DataAnalysisRequest, PredictionRequest, DocumentSearchRequest, DocumentQARequest,
    AnalysisResponse, PredictionResponse, DocumentSearchResponse, DocumentQAResponse
)

app = FastAPI(title="Intelligent Document Analysis API")
logger = setup_logger("api")

# Initialize components
data_loader = DataLoader()
document_processor = DocumentProcessor()
sales_predictor = SalesPredictor()
document_rag = DocumentRAG()

@app.get("/")
def root():
    """Root endpoint"""
    return {"message": "Welcome to the Intelligent Document Analysis API"}

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_data(request: DataAnalysisRequest):
    """Analyze data from a file"""
    try:
        # Load data
        df = data_loader.load_raw_data(request.file_path)
        
        # Clean data
        df_clean = document_processor.clean_data(df)
        
        # Transform features
        df_transformed = document_processor.transform_features(df_clean)
        
        # Detect outliers
        df_with_outliers = document_processor.detect_outliers(df_transformed)
        
        # Perform basic analysis
        analysis_results = {
            "record_count": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict(),
            "category_counts": {
                col: df[col].value_counts().to_dict() 
                for col in df.select_dtypes(include=['object', 'category']).columns
            },
            "outlier_count": df_with_outliers['is_outlier'].sum()
        }
        
        # If advanced analysis requested
        if request.analysis_type == "advanced":
            # Save the processed data for further use
            data_loader.save_processed_data(df_with_outliers, "processed_data.csv")
            
            # Additional analyses can be added here
            
        # If predictive analysis requested
        if request.analysis_type == "predictive" and request.target_column:
            # Train a model
            target = request.target_column
            metrics = sales_predictor.train(df_with_outliers, target_column=target)
            
            # Save the model
            sales_predictor

            # src/api/main.py (continued)
            # Save the model
            sales_predictor.save_model(f"{target}_predictor.pkl")
            
            # Add model metrics to results
            analysis_results["model_metrics"] = metrics
        
        return AnalysisResponse(
            success=True,
            message=f"Analysis completed successfully with {len(df)} records",
            results=analysis_results
        )
        
    except Exception as e:
        logger.error(f"Error in data analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make predictions using a trained model"""
    try:
        # Load the model
        model_loaded = sales_predictor.load_model(request.model_path)
        
        if not model_loaded:
            raise HTTPException(status_code=404, detail="Model not found or could not be loaded")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame(request.data)
        
        # Process input data
        input_df_processed = document_processor.transform_features(input_df)
        
        # Make predictions
        predictions = sales_predictor.predict(input_df_processed)
        
        if predictions is None:
            raise HTTPException(status_code=500, detail="Error making predictions")
        
        return PredictionResponse(
            success=True,
            message=f"Successfully generated {len(predictions)} predictions",
            predictions=predictions.tolist()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build-index", response_model=AnalysisResponse)
def build_document_index(request: DataAnalysisRequest, background_tasks: BackgroundTasks):
    """Build a document index for search and QA"""
    try:
        # Load data
        df = data_loader.load_processed_data(request.file_path)
        
        # Build index in background (for large datasets)
        def build_index_task():
            document_rag.build_index(df)
            document_rag.save_index()
        
        background_tasks.add_task(build_index_task)
        
        return AnalysisResponse(
            success=True,
            message="Document index building started in background",
            results={"status": "processing"}
        )
        
    except Exception as e:
        logger.error(f"Error building document index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=DocumentSearchResponse)
def search_documents(request: DocumentSearchRequest):
    """Search for similar documents"""
    try:
        # Load index if not already loaded
        if document_rag.vector_store is None:
            document_rag.load_index()
        
        # Search for documents
        results = document_rag.search_documents(request.query, k=request.top_k)
        
        # Format results
        formatted_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": i  # Use index as proxy for score (actual scores not easily accessible)
            }
            for i, doc in enumerate(results)
        ]
        
        return DocumentSearchResponse(
            success=True,
            message=f"Found {len(results)} matching documents",
            results=formatted_results
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=DocumentQAResponse)
def ask_question(request: DocumentQARequest):
    """Ask a question about the documents"""
    try:
        # Load index if not already loaded
        if document_rag.vector_store is None:
            document_rag.load_index()
        
        # Ask question
        answer = document_rag.ask(request.question)
        
        return DocumentQAResponse(
            success=True,
            message="Question answered successfully",
            answer=answer
        )
        
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """Start the API server"""
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)

if __name__ == "__main__":
    start()