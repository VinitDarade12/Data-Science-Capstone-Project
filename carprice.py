# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('CAR DETAILS.csv')

# Exploratory Data Analysis
st.title('Car Price Prediction App')

# View the first few rows
st.subheader('Dataset Overview')
st.write(df.head())

# Display basic information and statistics
st.subheader('Dataset Information')
st.write(df.info())
st.write(df.describe())

# Check for missing values
st.subheader('Missing Values')
st.write(df.isnull().sum())

# Plot Distribution of Selling Price
st.subheader('Price Distribution')
plt.figure(figsize=(8,6))
sns.histplot(df['selling_price'], kde=True)
plt.title('Price Distribution')
st.pyplot(plt)

# Plot boxplot for years vs. selling price
st.subheader('Top 10 Years vs. Selling Price')
top_years = df['year'].value_counts().nlargest(10).index
df_top_years = df[df['year'].isin(top_years)]
plt.figure(figsize=(8,6))
sns.boxplot(x='year', y='selling_price', data=df_top_years)
plt.title('Top 10 Years vs Selling Price')
plt.xticks(rotation=45, ha='right')
st.pyplot(plt)

# Plot Top 10 Car Brands Distribution
st.subheader('Top 10 Car Brands Distribution')
top_brands = df['name'].value_counts().nlargest(10).index
df_top_brands = df[df['name'].isin(top_brands)]
plt.figure(figsize=(10,6))
sns.countplot(x='name', data=df_top_brands, order=top_brands)
plt.xticks(rotation=45, ha='right')
st.pyplot(plt)

# Preprocessing (One-Hot Encoding & Scaling)
df = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()
df[['year', 'km_driven']] = scaler.fit_transform(df[['year', 'km_driven']])

# Define X and y
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Linear Regression Evaluation
st.subheader('Linear Regression Model Evaluation')
st.write("Linear Regression RMSE:", mean_squared_error(y_test, y_pred, squared=False))
st.write("R-squared:", r2_score(y_test, y_pred))

# Train Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Random Forest Evaluation
st.subheader('Random Forest Model Evaluation')
st.write("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
st.write("R-squared:", r2_score(y_test, y_pred_rf))

# Save the best model (Random Forest in this case)
with open('best_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Load the model for predictions
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit User Input for Car Price Prediction
st.title('Car Price Prediction')

# Collect user input
year = st.number_input('Year (Scaled)', min_value=float(df['year'].min()), max_value=float(df['year'].max()))
km_driven = st.number_input('Kilometers Driven (Scaled)', min_value=float(df['km_driven'].min()), max_value=float(df['km_driven'].max()))

# Prediction
if st.button('Predict'):
    prediction = model.predict([[year, km_driven]])
    st.success(f'The predicted car price is: ₹{prediction[0]:,.2f}')
