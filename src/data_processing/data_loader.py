# src/data_processing/data_loader.py
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def load_csv(self, file_path):
        """Load CSV file with error handling"""
        try:
            return pd.read_csv(Path(self.config['data_dir']) / file_path)
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {file_path}")
            raise e


