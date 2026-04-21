import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Define date range
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
num_days = (end_date - start_date).days
dates = [start_date + timedelta(days=i) for i in range(num_days)]

# Define possible values for each column
regions = ['North', 'South', 'East', 'West', 'Central']
products = ['Laptop', 'Keyboard', 'Mouse', 'Monitor', 'Webcam', 'Headphones']
categories = ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics']

quantities = np.random.randint(1, 5, size=len(dates))
unit_prices = np.random.uniform(50, 500, size=len(dates)).round(2)

# Create lists for each column
order_ids = [f'ORD{i+1:04d}' for i in range(len(dates))]
selected_dates = np.random.choice(dates, size=len(dates), replace=False)
regions_data = np.random.choice(regions, size=len(dates))
products_data = np.random.choice(products, size=len(dates))
categories_data = [categories[products.index(prod)] for prod in products_data]
total_sales = (quantities * unit_prices).round(2)

# Create a Pandas DataFrame
data = pd.DataFrame({
    'OrderID': order_ids,
    'Date': selected_dates,
    'Region': regions_data,
    'Product': products_data,
    'Category': categories_data,
    'Quantity': quantities,
    'UnitPrice': unit_prices,
    'TotalSale': total_sales
})

# Save the DataFrame to a CSV file
data.to_csv('sales_data.csv', index=False)

print("sales_data.csv has been created successfully!")
