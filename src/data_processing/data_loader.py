# src/data_processing/data_loader.py
import pandas as pd
import os
from src.utils.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

class DataLoader:
    """Class to handle data loading operations"""
    
    @staticmethod
    def load_raw_data(filename):
        """Load raw data from the raw data directory"""
        file_path = os.path.join(RAW_DATA_DIR, filename)
        logger.info(f"Loading raw data from {file_path}")
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif filename.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
            
            logger.info(f"Successfully loaded {len(df)} records")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    @staticmethod
    def save_interim_data(df, filename):
        """Save data to the interim directory"""
        file_path = os.path.join(INTERIM_DATA_DIR, filename)
        logger.info(f"Saving interim data to {file_path}")
        
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Successfully saved {len(df)} records to {file_path}")
        except Exception as e:
            logger.error(f"Error saving interim data: {str(e)}")
            raise
    
    @staticmethod
    def save_processed_data(df, filename):
        """Save data to the processed directory"""
        file_path = os.path.join(PROCESSED_DATA_DIR, filename)
        logger.info(f"Saving processed data to {file_path}")
        
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Successfully saved {len(df)} records to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    @staticmethod
    def load_processed_data(filename):
        """Load processed data from the processed data directory"""
        file_path = os.path.join(PROCESSED_DATA_DIR, filename)
        logger.info(f"Loading processed data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} processed records")
            return df
        except Exception as e:
            logger.error(f"Error loading processed data: {str(e)}")
            raise