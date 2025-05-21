# src/api/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DataAnalysisRequest(BaseModel):
    """Schema for data analysis request"""
    file_path: str
    analysis_type: str = "basic"  # Options: "basic", "advanced", "predictive"
    target_column: Optional[str] = None
    
class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    model_path: str
    data: List[Dict[str, Any]]
    
class DocumentSearchRequest(BaseModel):
    """Schema for document search request"""
    query: str
    top_k: int = 5
    
class DocumentQARequest(BaseModel):
    """Schema for document QA request"""
    question: str
    
class AnalysisResponse(BaseModel):
    """Schema for analysis response"""
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    
class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    success: bool
    message: str
    predictions: Optional[List[float]] = None
    
class DocumentSearchResponse(BaseModel):
    """Schema for document search response"""
    success: bool
    message: str
    results: Optional[List[Dict[str, Any]]] = None
    
class DocumentQAResponse(BaseModel):
    """Schema for document QA response"""
    success: bool
    message: str
    answer: Optional[str] = None