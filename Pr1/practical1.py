import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv(r"D:\SEM-3 Practicals\Machine-Learning-Practicals-Sem3-150\Pr1\attendance.csv")
# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Replace missing values with NaN
df.replace("", np.nan, inplace=True)

# Count missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill numerical columns with mean
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Verify dataset
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())