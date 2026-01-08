import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/33684/Data_science_project/dataset/class_ny_arrests.csv"
df = pd.read_csv(file_path)

# 1. Data Dimensionality
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 2. Data Granularity
print("Sample Data:\n", df.head())

# 3. Data Distribution
# Numerical columns
df.describe().T

# Plot histograms for numerical features
df.hist(figsize=(10, 10))
plt.show()

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nValue Counts for {col}:\n", df[col].value_counts())

# 4. Data Sparsity
missing_data = df.isnull().sum()
print("\nMissing Data:\n", missing_data[missing_data > 0])

# Identify columns with >90% missing
sparse_cols = missing_data[missing_data / len(df) > 0.9].index
print("\nSparse Columns:\n", sparse_cols)

# Outliers using boxplots
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()
