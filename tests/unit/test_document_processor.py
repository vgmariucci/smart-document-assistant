# tests/unit/test_document_processor.py
import pytest
import pandas as pd
import numpy as np
from src.data_processing.document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Unit tests for the DocumentProcessor class"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.processor = DocumentProcessor()
        
        # Create a small test dataframe with intentional issues
        self.test_df = pd.DataFrame({
            'transaction_id': ['TXN1001', 'TXN1002', 'TXN1002', 'TXN1003'],  # Note: duplicate
            'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'product_category': ['Laptops', 'Smartphones', 'Accessories', 'Tablets'],
            'unit_price': [1200.0, 800.0, 50.0, np.nan],  # Note: missing value
            'quantity': [1, 2, 3, 1],
            'customer_email': ['user1@example.com', None, 'user3@example.com', 'user4@example.com'],  # Note: None
            'total_price': [1200.0, 1600.0, 200.0, 500.0]  # Note: 3rd value should be 150.0 (50*3)
        })
        
    def test_clean_data(self):
        """Test data cleaning functionality"""
        df_clean = self.processor.clean_data(self.test_df)
        
        # Check duplicate removal
        assert len(df_clean) == 3
        
        # Check data types
        assert pd.api.types.is_datetime64_dtype(df_clean['transaction_date'])
        assert pd.api.types.is_float_dtype(df_clean['unit_price'])
        assert pd.api.types.is_integer_dtype(df_clean['quantity'])
        
        # Check that missing values are still present (we didn't remove them)
        assert df_clean['unit_price'].isna().sum() == 1
        
    def test_transform_features(self):
        """Test feature transformation"""
        # First clean the data
        df_clean = self.processor.clean_data(self.test_df)
        
        # Then transform features
        df_transformed = self.processor.transform_features(df_clean)
        
        # Check new date features were created
        assert 'purchase_year' in df_transformed.columns
        assert 'purchase_month' in df_transformed.columns
        assert 'purchase_dayofweek' in df_transformed.columns
        
        # Check new binary flags
        assert 'has_email' in df_transformed.columns
        assert df_transformed.loc[df_transformed['customer_email'].isna(), 'has_email'].iloc[0] == 0
        assert df_transformed.loc[~df_transformed['customer_email'].isna(), 'has_email'].iloc[0] == 1
        
    def test_detect_outliers(self):
        """Test outlier detection"""
        # Create a dataframe with outliers
        outlier_df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50, 1000]  # 1000 is an outlier
        })
        
        # Detect outliers
        result_df = self.processor.detect_outliers(outlier_df, columns=['value'])
        
        # Check outlier flag
        assert 'is_outlier' in result_df.columns
        assert result_df['is_outlier'].sum() == 1
        assert result_df.loc[result_df['value'] == 1000, 'is_outlier'].iloc[0] == 1