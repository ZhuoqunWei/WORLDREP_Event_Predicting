import pandas as pd

# Load the parquet file
daily = pd.read_parquet('artifacts/daily_dataset.parquet')

# View the first few rows
print(daily.head())

# Check the column names and data types
print(daily.info())

# Get summary statistics
print(daily.describe())

# Check for missing values
print(daily.isnull().sum())

# See unique values in key columns (like 'split' if present)
if 'split' in daily.columns:
    print(daily['split'].unique())

# Check the date range
print(daily['date'].min(), daily['date'].max())

# Check available countries
print(daily['country'].unique())

import matplotlib.pyplot as plt

# Sum counts per country
country_totals = daily.groupby('country')['count'].sum().sort_values(ascending=False).head(10)
country_totals.plot(kind='bar')
plt.title('Top 10 Countries by Total Protest Events')
plt.ylabel('Total Event Count')
plt.show()
