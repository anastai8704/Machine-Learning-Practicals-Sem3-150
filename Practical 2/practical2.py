# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:36:20 2026

@author: Lenovo
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Load cleaned dataset
df = pd.read_csv("attendance_cleaned.csv")

# 2. Display first 5 rows
print("First 5 rows:")
print(df.head())

# 3. Define Feature (X) and Target (Y)
X = df[["Total_Absent"]]
y = df["Total_Present"]

# 4. Create Linear Regression model
model = LinearRegression()

# 5. Train the model
model.fit(X, y)

# 6. Predict Total_Present
y_pred = model.predict(X)

# 7. Display regression details
print("\nCoefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# 8. Calculate R-squared
r2 = r2_score(y, y_pred)
print("R-squared:", r2)

# 9. Display predictions
print("\nPredicted Total_Present:")
print(y_pred)

# 10. Plot actual data points
plt.scatter(X, y)

# 11. Plot regression line
plt.plot(X, y_pred)

# 12. Add labels and title
plt.xlabel("Total Absent")
plt.ylabel("Total Present")
plt.title("Simple Linear Regression")

# 13. Display graph
plt.show()
