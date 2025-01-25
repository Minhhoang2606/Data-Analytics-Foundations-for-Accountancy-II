'''
Restaurant tips prediction using machine learning
Author: Henry Ha
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Tips dataset
tips_data = sns.load_dataset('tips')

# General information about the dataset
print(tips_data.head())

# Display dataset information
print(tips_data.info())

# Display summary statistics
print(tips_data.describe())

# Plot distributions of numerical features as subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(tips_data['total_bill'], kde=True, bins=20, color='blue', ax=axes[0])
axes[0].set_title('Distribution of Total Bill')
axes[0].set_xlabel('Total Bill ($)')

sns.histplot(tips_data['tip'], kde=True, bins=20, color='green', ax=axes[1])
axes[1].set_title('Distribution of Tips')
axes[1].set_xlabel('Tip Amount ($)')

plt.tight_layout()
plt.show()

# Plot counts of categorical features as subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(x='sex', data=tips_data, palette='pastel', ax=axes[0, 0])
axes[0, 0].set_title('Distribution by Gender')

sns.countplot(x='smoker', data=tips_data, palette='pastel', ax=axes[0, 1])
axes[0, 1].set_title('Smoker vs. Non-Smoker')

sns.countplot(x='day', data=tips_data, palette='pastel', order=['Thur', 'Fri', 'Sat', 'Sun'], ax=axes[1, 0])
axes[1, 0].set_title('Distribution by Day of the Week')

sns.countplot(x='time', data=tips_data, palette='pastel', ax=axes[1, 1])
axes[1, 1].set_title('Lunch vs. Dinner')

plt.tight_layout()
plt.show()

# Plot a heatmap of correlations
sns.heatmap(tips_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Interaction Between Features

# Present these plots as 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(x='total_bill', y='tip', data=tips_data, hue='time', palette='coolwarm', ax=axes[0])
axes[0].set_title('Total Bill vs Tip')

sns.boxplot(x='day', y='tip', data=tips_data, palette='pastel', order=['Thur', 'Fri', 'Sat', 'Sun'], ax=axes[1])
axes[1].set_title('Tips by Day')

plt.tight_layout()
plt.show()


#TODO Data preprocessing

# One-hot encode categorical features
tips_data = pd.get_dummies(tips_data, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)
print(tips_data.head())

from sklearn.preprocessing import MinMaxScaler

# Apply Min-Max scaling
scaler = MinMaxScaler()
tips_data[['total_bill', 'tip', 'size']] = scaler.fit_transform(tips_data[['total_bill', 'tip', 'size']])

from sklearn.model_selection import train_test_split

# Define features and target variable
X = tips_data.drop(columns=['tip'])
y = tips_data['tip']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TODO Regression Model Implementation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Fit the model on the training data
linear_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_linear = linear_model.predict(X_test)

# Evaluate the model
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

print(f"Linear Regression Mean Squared Error: {linear_mse}")
print(f"Linear Regression R-squared: {linear_r2}")

from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train the Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions on the test data
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate the model
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_r2 = r2_score(y_test, y_pred_poly)

print(f"Polynomial Regression Mean Squared Error: {poly_mse}")
print(f"Polynomial Regression R-squared: {poly_r2}")

#TODO Model evaluation

# Residual Plot for Linear Regression
residuals = y_test - y_pred_linear
plt.scatter(y_pred_linear, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot for Linear Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Residual Plot for Polynomial Regression
residuals_poly = y_test - y_pred_poly
plt.scatter(y_pred_poly, residuals_poly)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot for Polynomial Regression')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

#TODO Addressing Underfitting in the Linear Regression Mod

# Add interaction term between total_bill and size
tips_data['total_bill_size'] = tips_data['total_bill'] * tips_data['size']

# Add squared terms for total_bill and size
tips_data['total_bill_squared'] = tips_data['total_bill'] ** 2
tips_data['size_squared'] = tips_data['size'] ** 2

import numpy as np

# Apply log transformation to total_bill and tip
tips_data['log_total_bill'] = np.log(tips_data['total_bill'] + 1)  # Avoid log(0)
tips_data['log_tip'] = np.log(tips_data['tip'] + 1)

# Define the new feature set
X = tips_data[['total_bill', 'size', 'total_bill_size', 'total_bill_squared', 'size_squared']]
y = tips_data['tip']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = linear_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Improved Linear Regression - MSE: {mse}, R-squared: {r2}")

import pickle

# Save the trained model
with open('linear_model.pkl', 'wb') as file:
    pickle.dump(linear_model, file)
