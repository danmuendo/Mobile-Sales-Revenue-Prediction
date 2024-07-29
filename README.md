# Mobile Sales Revenue Prediction

### Overview
This project aims to predict the total revenue for a given mobile model based on historical sales data using machine learning. The project involves data preprocessing, model training, and deploying a prediction application using Streamlit.

### Project Structure
app.py: Streamlit application code for making predictions.
model_training.py: Script for training the machine learning model.
random_forest_model.pkl: Trained Random Forest model.
scaler.pkl: Fitted StandardScaler for preprocessing.
feature_names.pkl: Saved feature names to ensure consistency between training and prediction.
mobile_sales.csv: Dataset containing historical sales data.

### Requirements
Python 3.7 or higher
Required Python packages (install via requirements.txt)

#### Install Required Packages

pip install -r requirements.txt


### Dataset

The dataset mobile_sales.csv contains the following columns:

TransactionID
Date
MobileModel
Brand
Price
UnitsSold
TotalRevenue
CustomerAge
CustomerGender
Location
PaymentMethod

### Model Training
Preprocess the Data: Convert date to datetime, extract year and month, and apply one-hot encoding to categorical variables.
Fit Scaler: Standardize numerical features.
Train Model: Train a Random Forest Regressor to predict TotalRevenue.

### Conclusion
This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model training to deploying an interactive prediction application using Streamlit. Follow the steps in this README to set up and run the project on your local machine.