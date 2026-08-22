import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -------------------------------------------------
# 1. Load and Explore Data
# -------------------------------------------------

invest_df = pd.read_csv("Invest2Profit.csv")
position_df = pd.read_csv("Position_Salaries.csv")
study_df = pd.read_csv("Study_Data.csv")

print("Invest2Profit Dataset:")
print(invest_df.head())

print("\nPosition Salaries Dataset:")
print(position_df.head())

print("\nStudy Data Dataset:")
print(study_df.head())


# -------------------------------------------------
# 2. Multiple Linear Regression
# Invest2Profit.csv
# -------------------------------------------------

X = invest_df[["R&D Spend", "Administration", "Marketing Spend"]]
y = invest_df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)


# -------------------------------------------------
# 3. Polynomial Regression
# Position_Salaries.csv
# -------------------------------------------------

X_poly = position_df[["Level"]]
y_poly = position_df["Salary"]

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y_poly, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_poly)
X_test_poly = poly.transform(X_test_poly)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train_poly)

poly_pred = poly_model.predict(X_test_poly)


# -------------------------------------------------
# 4. Decision Tree Regression
# Invest2Profit.csv
# -------------------------------------------------

tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)


# -------------------------------------------------
# 5. Random Forest Regression
# Invest2Profit.csv
# -------------------------------------------------

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_pred = forest_model.predict(X_test)


# -------------------------------------------------
# 6. Evaluation Function
# -------------------------------------------------

def evaluate_model(name, actual, predicted):

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)

    print("\n" + name)
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("MSE:", mse)


# -------------------------------------------------
# 7. Display Model Results
# -------------------------------------------------

evaluate_model(
    "Multiple Linear Regression",
    y_test,
    linear_pred
)

evaluate_model(
    "Polynomial Regression",
    y_test_poly,
    poly_pred
)

evaluate_model(
    "Decision Tree Regression",
    y_test,
    tree_pred
)

evaluate_model(
    "Random Forest Regression",
    y_test,
    forest_pred
)


# -------------------------------------------------
# 8. Feature Importance
# -------------------------------------------------

importance = pd.Series(
    forest_model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)