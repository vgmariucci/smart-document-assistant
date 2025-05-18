# scripts/generate_sample_data.py
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)

# Generate 100,000 rows
num_records = 100_000

# Date range for 2023
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Create base dataframe with non-dependent columns first
base_data = {
    "transaction_id": [f"TXN{1000000 + i}" for i in range(num_records)],
    "transaction_date": np.random.choice(dates, num_records),
    "product_category": np.random.choice(
        ["Laptops", "Smartphones", "Tablets", "Accessories", "Gaming Consoles"],
        num_records,
        p=[0.3, 0.25, 0.15, 0.2, 0.1]
    ),
    "quantity": np.random.randint(1, 5, num_records),
    "customer_id": [fake.uuid4()[:8] for _ in range(num_records)],
    "customer_type": np.random.choice(["Individual", "Business"], num_records, p=[0.85, 0.15]),
    "payment_method": np.random.choice(["Credit Card", "Debit Card", "Cash"], num_records),
    "warranty_years": np.random.choice([1, 2, 3], num_records, p=[0.6, 0.3, 0.1]),
    "store_location": [fake.city() for _ in range(num_records)]
}

# Create the base dataframe
df = pd.DataFrame(base_data)

# Now add the dependent columns
# Product brand based on product category
product_brands = {
    'Laptops': ["Dell", "HP", "Lenovo", "Apple", "Asus"],
    'Smartphones': ["Apple", "Samsung", "Google", "OnePlus"],
    'Tablets': ["Apple", "Samsung", "Lenovo", "Amazon"],
    'Accessories': ["Logitech", "Anker", "Belkin", "Generic"],
    'Gaming Consoles': ["Sony", "Microsoft", "Nintendo"]
}

# Add product brand based on product category
df['product_brand'] = df.apply(
    lambda row: np.random.choice(
        product_brands.get(row['product_category'], ["Generic"])
    ),
    axis=1
)

# Generate unit prices based on product category
def generate_price(category):
    if category == 'Laptops':
        return np.random.normal(1200, 300)
    elif category == 'Smartphones':
        return np.random.normal(800, 150)
    elif category == 'Tablets':
        return np.random.normal(500, 100)
    elif category == 'Gaming Consoles':
        return np.random.normal(400, 80)
    else:  # Accessories
        return np.random.normal(100, 50)

df['unit_price'] = df['product_category'].apply(generate_price)

# Calculate total price
df['total_price'] = df['unit_price'] * df['quantity']

# Round prices to 2 decimal places
df['unit_price'] = np.round(df['unit_price'], 2)
df['total_price'] = np.round(df['total_price'], 2)

# Add customer email with 5% null values
df['customer_email'] = [fake.email() if np.random.random() > 0.05 else None for _ in range(num_records)]

# Add 2% outliers
outlier_indices = np.random.choice(num_records, int(0.02*num_records))
df.loc[outlier_indices, 'unit_price'] *= 10

# Recalculate total price for outliers
df.loc[outlier_indices, 'total_price'] = df.loc[outlier_indices, 'unit_price'] * df.loc[outlier_indices, 'quantity']
df.loc[outlier_indices, 'total_price'] = np.round(df.loc[outlier_indices, 'total_price'], 2)

# Save to CSV
df.to_csv("data/raw/tech_sales_2023_2.csv", index=False)