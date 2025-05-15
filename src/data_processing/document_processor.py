# Example CSV/Excel processing
from src.data_processing.data_loader import DataLoader
from src.utils.config import load_config

config = load_config()
loader = DataLoader(config)

# Load business data
financial_data = loader.load_csv("financial_records.csv")

# Preprocessing pipeline
def clean_business_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw business data"""
    return (
        df.dropna(subset=['transaction_amount'])
        .assign(transaction_date=lambda x: pd.to_datetime(x['transaction_date']))
        .pipe(remove_outliers)
    )

cleaned_data = clean_business_data(financial_data)