# File: generate_sales_data.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Configure parameters
NUM_ROWS = 100_000
CATEGORIES = ['Laptops', 'Smartphones', 'Tablets', 'Accessories', 'Gaming']
BRANDS = ['Apple', 'Samsung', 'Dell', 'Sony', 'Microsoft']
MODELS = {
    'Apple': ['iPhone 15', 'MacBook Pro', 'iPad Air', 'Apple Watch'],
    'Samsung': ['Galaxy S23', 'Galaxy Tab', 'Buds Pro', 'Smart Monitor'],
    'Dell': ['XPS 15', 'Alienware', 'UltraSharp', 'Inspiron'],
    'Sony': ['PlayStation 5', 'WH-1000XM5', 'Bravia', 'Alpha A7'],
    'Microsoft': ['Surface Pro', 'Xbox Series X', 'Surface Laptop']
}

# Generate date range
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

data = {
    'transaction_id': [f'TX-{i:06}' for i in range(1, NUM_ROWS+1)],
    'date': [fake.date_between(start_date, end_date) for _ in range(NUM_ROWS)],
    'category': np.random.choice(CATEGORIES, NUM_ROWS, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'brand': np.random.choice(BRANDS, NUM_ROWS),
    'model': '',
    'price': np.round(np.random.uniform(50, 2500, NUM_ROWS), 2),
    'quantity': np.random.randint(1, 4, NUM_ROWS),
    'store_location': [fake.city() for _ in range(NUM_ROWS)],
    'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Gift Card'], NUM_ROWS),
    'customer_id': [fake.uuid4()[:8] for _ in range(NUM_ROWS)],
    'customer_segment': np.random.choice(['VIP', 'Regular', 'New'], NUM_ROWS, p=[0.1, 0.7, 0.2])
}

# Assign models based on brand
data['model'] = [np.random.choice(MODELS[row['brand']]) for _, row in pd.DataFrame(data).iterrows()]

df = pd.DataFrame(data)
df.to_csv('data/raw/tech_sales_2023_1.csv', index=False)