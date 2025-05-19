# src/data_processing/document_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger("document_processor")

class DocumentProcessor:
    """Class to handle document processing operations"""
    
    def __init__(self):
        """Initialize the document processor"""
        logger.info("Initializing document processor")
    
    def clean_data(self, df):
        """
        Clean the raw data:
        - Handle missing values
        - Remove duplicates
        - Fix data types
        """
        logger.info("Starting data cleaning process")
        
        # Make a copy to avoid modifying the original dataframe
        df_clean = df.copy()
        
        # Log initial statistics
        logger.info(f"Initial data shape: {df_clean.shape}")
        logger.info(f"Missing values: {df_clean.isnull().sum().sum()}")
        
        # Handle missing values
        logger.info("Handling missing values")
        # For customer email, we'll leave nulls as is since they might be genuine
        
        # Remove any duplicate transaction IDs
        logger.info("Removing duplicates based on transaction_id")
        df_clean = df_clean.drop_duplicates(subset=['transaction_id'])
        
        # Fix data types
        logger.info("Fixing data types")
        df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'])
        df_clean['unit_price'] = df_clean['unit_price'].astype(float)
        df_clean['quantity'] = df_clean['quantity'].astype(int)
        df_clean['total_price'] = df_clean['total_price'].astype(float)
        
        # Verify total_price calculation
        logger.info("Verifying total_price calculations")
        calculated_total = df_clean['unit_price'] * df_clean['quantity']
        price_discrepancy = (df_clean['total_price'] - calculated_total).abs() > 0.01
        
        if price_discrepancy.any():
            logger.warning(f"Found {price_discrepancy.sum()} records with total price discrepancies")
            logger.info("Correcting total_price values")
            df_clean['total_price'] = calculated_total
        
        # Log final statistics
        logger.info(f"Final data shape after cleaning: {df_clean.shape}")
        logger.info(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
        
        return df_clean
    
    def transform_features(self, df):
        """
        Transform features for analysis and modeling:
        - Extract date features
        - Create categorical encodings
        - Generate new features
        """
        logger.info("Starting feature transformation")
        
        # Make a copy to avoid modifying the original dataframe
        df_transformed = df.copy()
        
        # Extract date features
        logger.info("Extracting date features")
        df_transformed['purchase_year'] = df_transformed['transaction_date'].dt.year
        df_transformed['purchase_month'] = df_transformed['transaction_date'].dt.month
        df_transformed['purchase_day'] = df_transformed['transaction_date'].dt.day
        df_transformed['purchase_dayofweek'] = df_transformed['transaction_date'].dt.dayofweek
        df_transformed['purchase_quarter'] = df_transformed['transaction_date'].dt.quarter
        
        # Create binary flags
        logger.info("Creating binary flags")
        df_transformed['is_business'] = (df_transformed['customer_type'] == 'Business').astype(int)
        df_transformed['has_email'] = (~df_transformed['customer_email'].isna()).astype(int)
        
        # Create price categories
        logger.info("Creating price categories")
        df_transformed['price_category'] = pd.cut(
            df_transformed['unit_price'],
            bins=[0, 100, 500, 1000, 2000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Create purchase volume category
        logger.info("Creating purchase volume category")
        df_transformed['purchase_volume'] = pd.cut(
            df_transformed['quantity'],
            bins=[0, 1, 2, 3, float('inf')],
            labels=['Single', 'Double', 'Triple', 'Bulk']
        )
        
        logger.info(f"Feature transformation complete. New shape: {df_transformed.shape}")
        
        return df_transformed
    
    def detect_outliers(self, df, columns=['unit_price', 'total_price']):
        """
        Detect outliers in specified columns using IQR method
        """
        logger.info(f"Detecting outliers in columns: {columns}")
        
        outlier_indices = {}
        for column in columns:
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier bounds
                lower_bound = Q1 - (1.5 * IQR)
                upper_bound = Q3 + (1.5 * IQR)
                
                # Find outliers
                column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
                outlier_indices[column] = column_outliers
                
                logger.info(f"Found {len(column_outliers)} outliers in column '{column}'")
        
        # Get combined outlier indices
        all_outliers = set()
        for indices in outlier_indices.values():
            all_outliers.update(indices)
        
        logger.info(f"Total unique outliers across all columns: {len(all_outliers)}")
        
        # Create an outlier flag column
        df_with_flags = df.copy()
        df_with_flags['is_outlier'] = 0
        df_with_flags.loc[list(all_outliers), 'is_outlier'] = 1
        
        return df_with_flags