# tests/integration/test_data_pipeline.py
import os
import tempfile
import pandas as pd
from src.data_processing.data_loader import DataLoader
from src.data_processing.document_processor import DocumentProcessor
from src.models.classical_ml import SalesPredictor

class TestDataPipeline:
    """Integration tests for the data processing pipeline"""
    
    def setup_method(self):
        """Set up test environment"""
        self.data_loader = DataLoader()
        self.document_processor = DocumentProcessor()
        self.sales_predictor = SalesPredictor(model_type='random_forest')
        
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'transaction_id': [f'TXN{i+1000}' for i in range(100)],
            'transaction_date': pd.date_range(start='2023-01-01', periods=100),
            'product_category': pd.Series(['Laptops', 'Smartphones', 'Tablets']).sample(100, replace=True).reset_index(drop=True),
            'product_brand': pd.Series(['Apple', 'Samsung', 'Dell', 'HP']).sample(100, replace=True).reset_index(drop=True),
            'unit_price': [round(float(i * 10 + 500), 2) for i in range(100)],
            'quantity': [i % 5 + 1 for i in range(100)],
            'total_price': [round(float((i * 10 + 500) * (i % 5 + 1)), 2) for i in range(100)],
            'customer_type': pd.Series(['Individual', 'Business']).sample(100, replace=True).reset_index(drop=True),
            'payment_method': pd.Series(['Credit Card', 'Debit Card', 'Cash']).sample(100, replace=True).reset_index(drop=True),
            'warranty_years': [i % 3 + 1 for i in range(100)],
            'customer_email': [f'user{i}@example.com' if i % 10 != 0 else None for i in range(100)]
        })
        
        # Mock data directories
        # Save paths
        self.RAW_DATA_DIR = os.path.join(self.temp_dir.name, 'raw')
        self.INTERIM_DATA_DIR = os.path.join(self.temp_dir.name, 'interim')
        self.PROCESSED_DATA_DIR = os.path.join(self.temp_dir.name, 'processed')
        self.MODEL_DIR = os.path.join(self.temp_dir.name, 'models')
        
        # Create directories
        for directory in [self.RAW_DATA_DIR, self.INTERIM_DATA_DIR, self.PROCESSED_DATA_DIR, self.MODEL_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Save test file
        self.test_file = os.path.join(self.RAW_DATA_DIR, 'test_sales.csv')
        self.test_data.to_csv(self.test_file, index=False)
        
        # Override data directories in the classes
        self.original_raw_dir = DataLoader.RAW_DATA_DIR
        self.original_interim_dir = DataLoader.INTERIM_DATA_DIR
        self.original_processed_dir = DataLoader.PROCESSED_DATA_DIR
        self.original_model_dir = SalesPredictor.MODEL_DIR
        
        DataLoader.RAW_DATA_DIR = self.RAW_DATA_DIR
        DataLoader.INTERIM_DATA_DIR = self.INTERIM_DATA_DIR
        DataLoader.PROCESSED_DATA_DIR = self.PROCESSED_DATA_DIR
        SalesPredictor.MODEL_DIR = self.MODEL_DIR
        
    def teardown_method(self):
        """Clean up after test"""
        self.temp_dir.cleanup()
        
        # Restore original directories
        DataLoader.RAW_DATA_DIR = self.original_raw_dir
        DataLoader.INTERIM_DATA_DIR = self.original_interim_dir
        DataLoader.PROCESSED_DATA_DIR = self.original_processed_dir
        SalesPredictor.MODEL_DIR = self.original_model_dir
        
    def test_full_pipeline(self):
        """Test the full data processing and modeling pipeline"""
        # 1. Load data
        filename = os.path.basename(self.test_file)
        df = self.data_loader.load_raw_data(filename)
        assert len(df) == 100
        
        # 2. Clean data
        df_clean = self.document_processor.clean_data(df)
        assert len(df_clean) == 100  # No duplicates to remove
        
        # 3. Transform features
        df_transformed = self.document_processor.transform_features(df_clean)
        assert 'purchase_month' in df_transformed.columns
        assert 'has_email' in df_transformed.columns
        
        # 4. Detect outliers
        df_with_outliers = self.document_processor.detect_outliers(df_transformed)
        assert 'is_outlier' in df_with_outliers.columns
        
        # 5. Save processed data
        processed_filename = 'processed_test_sales.csv'
        self.data_loader.save_processed_data(df_with_outliers, processed_filename)
        assert os.path.exists(os.path.join(self.PROCESSED_DATA_DIR, processed_filename))
        
        # 6. Train model
        metrics = self.sales_predictor.train(df_with_outliers, target_column='total_price')
        assert 'rmse' in metrics
        assert 'r2' in metrics
        
        # 7. Save model
        model_saved = self.sales_predictor.save_model('test_model.pkl')
        assert model_saved
        assert os.path.exists(os.path.join(self.MODEL_DIR, 'test_model.pkl'))
        
        # 8. Load model 
        model_loaded = self.sales_predictor.load_model('test_model.pkl')
        assert model_loaded
        
        # 9. Make predictions
        predictions = self.sales_predictor.predict(df_with_outliers.head(5))
        assert len(predictions) == 5