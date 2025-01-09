## titanic Survival Prediction
#1. importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')
print(df)

# Exploring the dataset
print('First few rows:\n', df.head())
print('\nDataset Information:\n', df.info())
print('\nMissing Values:\n', df.isnull().sum())
print(df.describe())

#3. Exploratory Data Analysis (EDA)
# Survival Distribution
sns.countplot(data=df, x='Survived', palette='viridis')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1=Yes)')
plt.ylabel('Count')
plt.show()
# Survival Distribution shows the ratio of survivors vs. non-survivors.

# Survival by Gender
sns.countplot(data=df, x='Survived', hue='Sex', palette='coolwarm')
plt.title('Survival (0=No, 1=Yes)')
plt.ylabel('Count')
plt.legend(title='Gender', labels=['Male', 'Female'])
plt.show()
# Survival by Gender highlights that females had higher survival rates.

# Age Distribution by Survival
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30, palette='muted')
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()
# Age Distribution by Survival illustrates survival trends across different age groups.

# Survival by Passenger Class
sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set2')
plt.title('Survival by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
# Survival by Passenger Class revels that first-class passengers had better chances of survival.

'''Conclusion: Visualization helps uncover relationships and trends in the data, 
   providing insights that guide preprocessing and model building.  '''

# 4.Data Preprocessing
# Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Missing values in Age and Embarked are filled with median and mode, respectively.

# Drop columns with excessive missing values or irrelevant for prediction
df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)
# The Cabin column is dropped due to excessive missing data.

# 5.Encode categorical variables
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2})
# Categorical variables (Sex and Embarked) are encoded numerically.

# 6.Create new features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print(df['FamilySize'])
# A new feature (FamilySize) is created to indicate family connections

# Drop redundant features
df.drop(columns=['SibSp', 'Parch'], inplace=True)

# Cleaned dataset
print('\nPrepossed Dataset few rows:\n',df.head())
''' Conclusion: Preprocessing prepares the dataset for machine learning by 
     ensuring completeness, consistency and usability. '''
    
# 7.Split Data into Training and Testing Sets
''' The dataset is split into training (80%) and testing (20%) subsets. The training set is used
    to fit the model, while the esting set evaluates its performance. '''
# Separate features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)
# Conclusion: Splitting ensures that the model is evaluated on unseen data, avoiding overfitting.

# 8. Train the Model
# Train a random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# conclusion: The model learns patterns in the training data to predict survival.

# Evaluate the model
y_pred = model.predict(X_test)
print('\nY Prediction:\n',y_pred)

# Accuracy Score  (Measures overall prediction correctness.)
print('\nAccuracy:\n', accuracy_score(y_test, y_pred))

# Confusion Matrix
# Breaks down predictions into true positives, true negatives, false positives, and false negatives. 
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report  (Summarizes precision, recall, and F1-score.)
print('\nClassification Report:\n', classification_report(y_test, y_pred))
# Conclusion: Evaluation metrics and visualizations confirm the model's effectiveness and highlight areas for improvement.

# 8. Insights and Conclusion
# EDA Insights:
''' * Females had higher survival rates than males.
    * First-class passengers were more likely to survive than second-or third-class passengers.
    * Survival rates varied with age and family size.  
'''
# important Features: analysis reveals the most significant predictors of survival (e.g., Pclass, Sex, Age).
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
print("\nFeature Importances:\n", feature_importances)  

