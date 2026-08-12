# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:24:52 2026

@author: Lenovo
"""
import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("attendance.csv")

# 2. Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# 3. Display dataset information
print("\nDataset Information:")
df.info()

# 4. Identify missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 5. Fill missing numerical values with mean
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# 6. Fill missing Student_Name using mode
if "Student_Name" in df.columns:
    df["Student_Name"] = df["Student_Name"].fillna(
        df["Student_Name"].mode()[0]
    )

# 7. Identify duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# 8. Remove duplicate rows
df = df.drop_duplicates()

# 9. Validate after preprocessing
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nDataset information after cleaning:")
df.info()

# 10. Save cleaned dataset
df.to_csv("attendance_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")
