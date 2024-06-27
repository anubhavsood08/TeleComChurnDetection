import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd
import sklearn 
# Load the data
df_1 = pd.read_csv('first_telc.csv')

# Streamlit UI elements
st.title('Churn Prediction App')
st.write('Enter the details of the customer')

SeniorCitizen = st.number_input('SeniorCitizen', min_value=0.00, max_value=1.00)
MonthlyCharges = st.number_input('Monthly Charges')
TotalCharges = st.number_input('Total Charges:')
gender = st.selectbox('Gender', ('Male', 'Female'))
Partner = st.selectbox('Partner', ('Yes', 'No'))
Dependents = st.selectbox('Dependents', ('Yes', 'No'))
PhoneService = st.selectbox('Phone Service', ('Yes', 'No'))
MultipleLines = st.selectbox('MultipleLines', ('Yes', 'No', 'No phone service'))
InternetService = st.selectbox('InternetServices', ('No', 'DSL', 'Fiber optic'))
OnlineSecurity = st.selectbox('OnlineSecurity', ('Yes', 'No', 'No internet service'))
OnlineBackup = st.selectbox('OnlineBackup', ('Yes', 'No', 'No internet service'))
DeviceProtection = st.selectbox('DeviceProtection', ('Yes', 'No', 'No internet service'))
TechSupport = st.selectbox('TechSupport', ('Yes', 'No', 'No internet service'))
StreamingTV = st.selectbox('StreamingTV', ('Yes', 'No', 'No internet service'))
StreamingMovies = st.selectbox('StreamingMovies', ('Yes', 'No', 'No internet service'))
Contract = st.selectbox('Contract', ('One year', 'Month-to-month', 'Two year'))
PaperlessBilling = st.selectbox('PaperlessBilling', ('Yes', 'No'))
PaymentMethod = st.selectbox('PaymentMethod', ('Credit card (automatic)', 'Mailed check', 'Electronic check', 'Bank transfer (automatic)'))
tenure = st.number_input('Tenure', min_value=0, max_value=100)

# Load the model
model = pkl.load(open('model.pkl', 'rb'))

# Prepare the input data
data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService,
         MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
         StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure]]

# Create a new DataFrame
new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
                                     'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                     'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                     'PaymentMethod', 'tenure'])

# Concatenate with the existing DataFrame
df_2 = pd.concat([df_1, new_df], ignore_index=True)

# Group the tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

# Drop the 'tenure' column
df_2.drop(columns=['tenure'], axis=1, inplace=True)

# Convert categorical variables to dummy variables
df_2 = pd.get_dummies(df_2, drop_first=True)

# Ensure the new DataFrame has all the columns the model was trained on
model_features = model.feature_names_in_
missing_cols = set(model_features) - set(df_2.columns)
for col in missing_cols:
    df_2[col] = 0

# Ensure the order of columns matches the model's expected input
df_2 = df_2[model_features]

# Prediction
if st.button('Predict'):
    prediction = model.predict(df_2.tail(1))
    prediction_proba = model.predict_proba(df_2.tail(1))[:, 1]

    if prediction[0] == 1:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

    st.write(f'Churn Probability: {prediction_proba[0]:.2f}')
    st.write(f'Not Churn Probability: {1 - prediction_proba[0]:.2f}')
