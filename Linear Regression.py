
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from typing import final
from scipy.sparse import data
from numpy.core.numeric import full
from numpy.lib import polynomial
from joblib import dump

full_table = pd.read_csv("C:\\Users\\2702b\\OneDrive - Asia Pacific University\\Diploma\\Semester 4\\Introducton of Data Analytics\\Assignment\\Datasets\\full_table.csv")

data_modelling_table = full_table[["product_length_cm", "product_weight_g", "freight_value"]]
data_modelling_table[["Product Length (cm)", "Product Weight (g)", "Shipping fee"]] = data_modelling_table[["product_length_cm", "product_weight_g", "freight_value"]]
data_modelling_table = data_modelling_table.drop(["product_length_cm", "product_weight_g", "freight_value"], axis=1)
data_modelling_table[data_modelling_table["Product Weight (g)"] > 40000]

data_modelling_table = data_modelling_table.drop(data_modelling_table[data_modelling_table["Product Weight (g)"] > 40000].index)

X = data_modelling_table.drop(["Product Weight (g)", "Shipping fee"], axis=1)
y = data_modelling_table["Shipping fee"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
length_regression_model = LinearRegression()
length_regression_model.fit(X_train, y_train)
score = length_regression_model.score(X_test, y_test)
length_predictions = length_regression_model.predict(X_test)
length_rmse = np.sqrt(mean_squared_error(y_test, length_predictions))


final_length_model = LinearRegression()
final_length_model.fit(X, y)
final_length_predictions = final_length_model.predict(X)
dump(final_length_model, "final_length_model.joblib")


X = data_modelling_table.drop(["Product Length (cm)", "Shipping fee"], axis=1)
y = data_modelling_table["Shipping fee"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
weight_regression_model = LinearRegression()
weight_regression_model.fit(X_train, y_train)
weight_score = weight_regression_model.score(X_test, y_test)
weight_predictions = weight_regression_model.predict(X_test)
weight_rmse = np.sqrt(mean_squared_error(y_test, weight_predictions))


final_weight_model = LinearRegression()
final_weight_model.fit(X, y)
final_weight_predictions = final_weight_model.predict(X)
dump(final_weight_model, "final_weight_model.joblib")


data_modelling_table["Predicted Shipping fee base on Length"] = final_length_predictions
data_modelling_table["Predicted Shipping fee base on Weight"] = final_weight_predictions
data_modelling_table.to_csv("Linear Regression.csv")

print(f"Length RMSE: {length_rmse}")
print(f"Weight RMSE: {weight_rmse}")
print(f"Mean value of Shipping fee: {data_modelling_table['Shipping fee'].mean()}")
print(f"Length coefficients: {final_length_model.coef_}")
print(f"Weight coefficients: {final_weight_model.coef_}")
