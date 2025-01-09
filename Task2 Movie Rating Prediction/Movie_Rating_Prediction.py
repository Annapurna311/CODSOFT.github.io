# Task 2- Movie Rating Prediction

# 1.Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score

# 2.Load and Inspect Data
df = pd.read_csv('IMDB Movies India.csv', encoding='latin1')

# Dataset first look
print('First few rows:\n',df.head(5))
print('\nDataset Information:\n',df.info())

# 3.Data Cleaning
# Initial DataFrame inspection
print("\nMissing Values Before Cleaning:\n",df.isnull().sum())

# Checking Duplicate Rows
print("\nDuplicate Rows Before Cleaning:\n",df.duplicated().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Recheck for missing values
print("\nMissing Values After Dropping:\n",df.isnull().sum())

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Final shape of the DataFrame
print("Final Shape of the DataFrame:", df.shape)

# 4.Data Pre-Processing
# replacing the brackets from year column
df['Year'] = df['Year'].str.replace(r'[()]', '', regex=True).astype(int)

# Remove the min word from 'Duration' column and convert all values to numeric
df['Duration'] = pd.to_numeric(df['Duration'].str.replace(' min', ''))

# Splitting the genre by, to keep only unique genres and replacing the null values with mode
df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre')
df['Genre'].fillna(df['Genre'].mode()[0], inplace=True)

# Convert 'Votes' to numeric and replace the to keep only numeric part
df['Votes'] = pd.to_numeric(df['Votes'].str.replace(',', ''))

# Checking the dataset is there any null values present and data types of the feature present
print('\nDataset Information:\n',df.info())

# 5.Data Visualizing
# Here we have created a histogram over the years in the data
plt.hist(df['Year'], bins=30, density=True, alpha=0.7, color='blue')
plt.title('Distribution of Movies by Year')
plt.xlabel('Year')
plt.ylabel('Probability Density')
plt.show()

# Group data by Year and calculate the average rating
avg_rating_by_year = df.groupby(['Year', 'Genre'])['Rating'].mean().reset_index()
print(avg_rating_by_year.head(10))

# Get the top 10 genres by average rating
top_generes = avg_rating_by_year.groupby('Genre')['Rating'].mean().sort_values(ascending=False).head(10).index
print(top_generes)

# Filter the data to include only the top 3 genres
average_rating_by_year = avg_rating_by_year[avg_rating_by_year['Genre'].isin(top_generes)]
print(average_rating_by_year.head(10))

# Create the line plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=average_rating_by_year, x='Year', y='Rating', hue='Genre', marker='o')
plt.title('Average Rating by Genre Over Time')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

# This histogram shows the distribution of ratings and its probability density
plt.hist(df['Rating'], bins=30, density=True, alpha=0.7, color='blue')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Probability Density')
plt.show()

# This scatter plot shows the relationship between the duration of the movie and its rating
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Duration', y='Rating', hue='Genre')
plt.title('Duration vs Rating')
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.show()

# 6.Feature Engineering
# Drop the Name column because it doesn't impact the outcome
df.drop('Name', axis=1, inplace=True)

# Grouping the columns with their average ratings and then creating a new feature
genre_mean_rating = df.groupby('Genre')['Rating'].transform('mean')
df['Genre_mean_rating'] = genre_mean_rating

director_mean_rating = df.groupby('Director')['Rating'].transform('mean')
df['Director_encoded'] = director_mean_rating

actor1_mean_rating = df.groupby('Actor 1')['Rating'].transform('mean')
df['Actor1_encoded'] = actor1_mean_rating

actor2_mean_rating = df.groupby('Actor 2')['Rating'].transform('mean')
df['Actor2_encoded'] = actor2_mean_rating

actor3_mean_rating = df.groupby('Actor 3')['Rating'].transform('mean')
df['Actor3_encoded'] = actor3_mean_rating

# Display the updated DataFrame
print(df.head())

# 7.Split data
# Keeping the predictor and target variables
X = df[['Year', 'Votes', 'Duration', 'Genre_mean_rating', 'Director_encoded', 'Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded']]
y = df['Rating']

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Test shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")

# 8.Model Building
# Initializing and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Calculating evaluation metrics
print('Accuracy (R2 Score):', model.score(X_test, y_test))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))

# Handling potential errors for negative predictions in mean_squared_log_error
y_pred = [max(val, 0) for val in y_pred]  
print('Mean Squared Log Error:', mean_squared_log_error(y_test, y_pred))

print('R2 Score:', r2_score(y_test, y_pred))

# Model Testing
print(X.head())
print(y.head())

# For testing , we create a new dataframe with values close to the any of our existing data to evaluate
data = {'Year': [2019], 'Votes': [36], 'Duration': [111], 'Genre_mean_rating': [5.8], 'Director_encoded' : [4.5], 'Actor1_encoded' : [5.3], 'Actor2_encoded' : [4.5], 'Actor3_encoded' : [4.5]}
trail = pd.DataFrame(data)

# Predict the movie rating by entered data
rating = model.predict(trail)

# Displat the predicted rating
print('Predicted Rating:', rating[0])

