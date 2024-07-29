import streamlit as st
import joblib
import pandas as pd

# Load the trained model, fitted scaler, and feature names
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Function to preprocess input data
def preprocess_data(input_data):
    # Convert the input data to DataFrame
    data_df = pd.DataFrame([input_data])

    # One-hot encode the input data
    data_df = pd.get_dummies(data_df)

    # Add missing columns with default value 0
    for col in feature_names:
        if col not in data_df.columns:
            data_df[col] = 0

    # Ensure the order of columns matches the training data
    data_df = data_df[feature_names]

    # Apply scaling to numerical features
    numeric_features = ['Price', 'UnitsSold', 'CustomerAge', 'Year', 'Month']
    data_df[numeric_features] = scaler.transform(data_df[numeric_features])

    return data_df

# Streamlit app
st.title('Mobile Sales Revenue Prediction')

# Input fields for user to enter data
price = st.number_input('Price', min_value=0.0, value=500.0)
units_sold = st.number_input('Units Sold', min_value=0, value=50)
customer_age = st.number_input('Customer Age', min_value=0, value=30)
year = st.number_input('Year', min_value=2020, value=2024)
month = st.number_input('Month', min_value=1, max_value=12, value=7)
mobile_model = st.selectbox('Mobile Model', options=['model1', 'model2', 'model3'])
brand = st.selectbox('Brand', options=['brand1', 'brand2', 'brand3'])
customer_gender = st.selectbox('Customer Gender', options=['Male', 'Female', 'Other'])
location = st.selectbox('Location', options=['loc1', 'loc2', 'loc3'])
payment_method = st.selectbox('Payment Method', options=['Online', 'Credit Card', 'Cash'])

# Create input data dictionary
input_data = {
    'Price': price,
    'UnitsSold': units_sold,
    'CustomerAge': customer_age,
    'Year': year,
    'Month': month,
    f'MobileModel_{mobile_model}': 1,
    f'Brand_{brand}': 1,
    f'CustomerGender_{customer_gender}': 1,
    f'Location_{location}': 1,
    f'PaymentMethod_{payment_method}': 1
}

# Preprocess and predict
if st.button('Predict'):
    input_df = preprocess_data(input_data)
    prediction = model.predict(input_df)
    st.write(f'Predicted Total Revenue: ${prediction[0]:.2f}')
