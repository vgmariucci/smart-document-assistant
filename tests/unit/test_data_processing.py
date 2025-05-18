# tests/unit/test_data_processing.py
def test_data_loading():
    test_data = pd.DataFrame({
        'transaction_date': ['2023-01-01', 'invalid_date'],
        'amount': [100, -50]
    })
    
    with pytest.raises(ValueError):
        clean_business_data(test_data)