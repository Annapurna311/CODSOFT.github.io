## Sales Prediction Using Python

# 1. Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load and Inspect Data
df = pd.read_csv('advertising.csv')
print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

# 3. Data Cleaning
# Check for Duplicates
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Drop duplicates if any
df = df.drop_duplicates()

# 4. Exploratory Data Analysis (EDA)
# Visualizing the relationships between features
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='scatter')
plt.show()

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Histograms of numerical features
df.hist(bins=30, figsize=(12, 10), color='blue', edgecolor='black')
plt.suptitle('Histograms of Features', y=1.02)
plt.show()

# 5. Feature Selection and Splitting
# Define the predictor variables (X) and the target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]  # Independent variables
y = df['Sales']  # Dependent variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 6. Model Building and Training
# Initialize the model
model = LinearRegression()

# Fit the model on training data
model.fit(X_train, y_train)

# 7. Model Evaluation
# Predictions on testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 8. Visualization of Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid()
plt.show()

# 9. Coefficients and Insights
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("Feature Coefficients:")
print(coefficients)

# Insights
# Higher coefficients indicate stronger influence on sales.
# TV and Radio usually have higher coefficients compared to Newspaper.
# TV has the highest positive impact on sales, followed by Radio.

