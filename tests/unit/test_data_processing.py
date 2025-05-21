# tests/unit/test_data_loader.py
import os
import pytest
import pandas as pd
import tempfile
from src.data_processing.data_loader import DataLoader

class TestDataLoader:
    """Unit tests for the DataLoader class"""
    
    def setup_method(self):
        """Set up test environment before each test"""
        self.data_loader = DataLoader()
        
        # Create a temporary file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a small test dataframe
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10, 20, 30]
        })
        
        # Save test file
        self.test_csv = os.path.join(self.temp_dir.name, 'test.csv')
        self.test_df.to_csv(self.test_csv, index=False)
        
        # Mock raw data directory for testing
        self.original_raw_dir = DataLoader.RAW_DATA_DIR
        DataLoader.RAW_DATA_DIR = self.temp_dir.name
        
    def teardown_method(self):
        """Clean up after each test"""
        self.temp_dir.cleanup()
        # Restore original data directory
        DataLoader.RAW_DATA_DIR = self.original_raw_dir
        
    def test_load_raw_data(self):
        """Test loading raw data from CSV"""
        # Get the filename part only
        filename = os.path.basename(self.test_csv)
        df = self.data_loader.load_raw_data(filename)
        
        # Check that the correct data was loaded
        assert len(df) == 3
        assert set(df.columns) == {'id', 'name', 'value'}
        assert df['id'].tolist() == [1, 2, 3]
        
    def test_load_raw_data_unsupported_format(self):
        """Test error handling for unsupported file formats"""
        with tempfile.NamedTemporaryFile(suffix='.txt', dir=self.temp_dir.name) as temp_file:
            filename = os.path.basename(temp_file.name)
            with pytest.raises(ValueError, match="Unsupported file format"):
                self.data_loader.load_raw_data(filename)

