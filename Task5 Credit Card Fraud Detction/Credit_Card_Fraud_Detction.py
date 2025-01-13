## CREDIT CARD FRAUD DETECTION

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 2. Load and Inspect Data
df = pd.read_csv('creditcard.csv')
print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

# Value counts of the class column
print(df['Class'].value_counts())

# 3. Exploratory Data Analysis(EDA)
# Visualizing the Class Column (Target Variable)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Class', palette='viridis')
plt.title('Class Distribution')
plt.xlabel('Class (0: Genuine, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Matrix')
plt.show()
# A red cell might indicate a strong positive correlation between "Transaction Amount" and "Transaction Type."
# A blue cell might show a strong negative correlation between "Account Age" and "Fraud Risk."

# Checking the balance of the dataset
fraud_count = df['Class'].value_counts()
fraud_rate = 100*fraud_count/df.shape[0]
fraud_data = pd.concat([fraud_count, fraud_rate], axis=1).reset_index()
fraud_data.columns = ['Class', 'Count', 'Rate']
print(fraud_data)

# 4. Data Preprocessing
# Handling Class Imbalance
df_fraud = df[df['Class'] == 1]
df_not_fraud = df[df['Class'] == 0]
df_not_fraud_sampled = df_not_fraud.sample(df_fraud.shape[0], replace=False, random_state=101)
df_balanced = pd.concat([df_fraud, df_not_fraud_sampled], axis=0)
print(df_balanced)

# Checking the balance of the dataset
fraud_count = df_balanced['Class'].value_counts()
fraud_rate = 100*fraud_count/df_balanced.shape[0]
fraud_data = pd.concat([fraud_count, fraud_rate], axis=1).reset_index()
fraud_data.columns = ['Class', 'Count', 'Rate']
print(fraud_data)

# 5. Feature Selection and Splitting
# Standardizing the feature values
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(columns=['Amount'], inplace=True)

# Splitting the Balanced Data
X_balanced = df_balanced.drop(columns=['Class'])
y_balanced = df_balanced['Class']

# Standardizing the feature values for balanced data
X_balanced_scaled = scaler.fit_transform(X_balanced)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced_scaled, y_balanced, test_size=0.3, random_state=42)

# 6. Model Building
# Building the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 7. Model Evaluation
# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Fraudulent'], yticklabels=['Genuine', 'Fraudulent'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8. Feature Importance
# Identify the most important features
feature_importance = pd.Series(model.feature_importances_, index=X_balanced.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
