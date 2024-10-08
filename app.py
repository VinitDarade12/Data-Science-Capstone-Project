import pandas as pd
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')

# View the first few rows
df.head()
# Check for missing values and data types
df.info()

# Descriptive statistics
df.describe()
# Check for missing values
df.isnull().sum()
# Check the column names
print(df.columns)
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of target variable (e.g., Price)
plt.figure(figsize=(8,6))
sns.histplot(df['selling_price'], kde=True)
plt.title('Price Distribution')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Get the top N most common years
top_years = df['year'].value_counts().nlargest(10).index

# Filter the dataframe to include only the top N years
df_top_years = df[df['year'].isin(top_years)]

# Create the boxplot with the top N years
plt.figure(figsize=(8,6))
sns.boxplot(x='year', y='selling_price', data=df_top_years)
plt.title('Top 10 Years vs Selling Price')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.xticks(rotation=45, ha='right')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Get the top 10 car brands by count
top_brands = df['name'].value_counts().nlargest(10).index

# Filter the DataFrame to include only the top 10 brands
df_top_brands = df[df['name'].isin(top_brands)]

# Create the bar plot for the top 10 car brands
plt.figure(figsize=(10,6))
sns.countplot(x='name', data=df_top_brands, order=top_brands)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Car Brands Distribution')
plt.xlabel('Car Brands')
plt.ylabel('Count')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Get the top 10 car brands by count
top_brands = df['name'].value_counts().nlargest(10).index

# Filter the DataFrame to include only the top 10 brands
df_top_brands = df[df['name'].isin(top_brands)]

# Create the bar plot for the top 10 car brands
plt.figure(figsize=(10,6))
sns.countplot(x='name', data=df_top_brands, order=top_brands)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Car Brands Distribution')
plt.xlabel('Car Brands')
plt.ylabel('Count')
plt.show()
# One-Hot Encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# Example: Scaling numeric features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['year', 'km_driven']] = scaler.fit_transform(df[['year', 'km_driven']])

# Train-Test Split:

from sklearn.model_selection import train_test_split

X = df.drop('selling_price', axis=1)  # Features
y = df['selling_price']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Machine Learning Models:

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Model Evaluation
print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R-squared:", r2_score(y_test, y_pred))

# Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Model Evaluation
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("R-squared:", r2_score(y_test, y_pred_rf))
import pickle

# Save the model
import pickle
import streamlit as st

# Save the model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Load the model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Car Price Prediction App')

# Collect user input
st.title('Car Price Prediction App')

# Collect user input
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Assuming df is your original DataFrame that you used to train the model
df = pd.read_csv('CAR DETAILS.csv')

# One-Hot Encoding for categorical variables (to match the model training)
df = pd.get_dummies(df, drop_first=True)

# Get the list of columns in the trained model
model_columns = df.drop('selling_price', axis=1).columns

st.title('Car Price Prediction App')

# Collect user input
age = st.number_input('Car Age', min_value=0)
year = st.number_input('Year of Manufacture', min_value=1900, max_value=2024)
km_driven = st.number_input('Kilometers Driven', min_value=0)

# Create a DataFrame for the input
input_data = {
    'age': [age],
    'year': [year],
    'km_driven': [km_driven]
}

# Create a DataFrame from the input
input_df = pd.DataFrame(input_data)

# One-Hot Encoding for input features (to match model training)
input_df = pd.get_dummies(input_df, drop_first=True)

# Reindex the input DataFrame to ensure it has the same columns as the model
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.success(f'The predicted car price is: {prediction[0]}')
