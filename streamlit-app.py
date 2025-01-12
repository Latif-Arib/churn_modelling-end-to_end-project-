import streamlit as st
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle


# load the models
with open('models/CatBoostClassifier.pkl', 'rb') as model:
    ml_model = pickle.load(model)

dl_model = load_model('models/ann_model.h5')   

# load the label encoder
with open('encoder/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# load the one hot encoder
with open('encoder/one_hot_encoder.pkl', 'rb') as file:
    ohe = pickle.load(file)

# load the scaler
with open('encoder/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Churn Prediction App')


# def calculate_shap(model, X_train, X_test):
#     """
#     Calculate SHAP values for the model
#     Args:
#     model: Model object
#     X_train: Training data
#     X_test: Testing data
#     Returns:
#     shap_values_train: SHAP values for training data    
#     shap_values_test: SHAP values for testing data
#     explainer: SHAP explainer object
#     """
#     explainer = shap.TreeExplainer(model)
#     shap_values_train = explainer.shap_values(X_train)
#     shap_values_test = explainer.shap_values(X_test)
#     return explainer, shap_values_train, shap_values_test

# def plot_shap_summary(explainer, shap_values, X_train):
#     """
#     Plot SHAP summary plot
#     Args:
#     explainer: SHAP explainer object
#     shap_values: SHAP values
#     X_train: Training data
#     """
#     shap.summary_plot(shap_values, X_train, plot_type='bar')
#     plt.show()

# Radio button to select the service
service = st.radio('Which Model you want to do the prediction?',['ML Model','ANN Model'] )

# if service == "To Know the model's feature importance":
#     # calculate SHAP values
#     explainer, shap_values_train, shap_values_test = calculate_shap(model, X_train, X_test)
#     plot_shap_summary(explainer, shap_values_train, X_train)

# user input
credit_score = st.number_input('Credit Score')
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.radio('Has Credit Card', [0,1])
is_active_member = st.radio('Is Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary')

# make a DataFrame from inputs
input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender':[gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }
)

# Tranasform the input data
geography_encoded = ohe.transform(input_data[['Geography']])
geography_encoded_df = pd.DataFrame(geography_encoded.toarray(), columns=ohe.get_feature_names_out(['Geography']))

# Concatenate the DataFrame with the encoded columns
input_data = pd.concat([geography_encoded_df,input_data], axis=1)
input_data.drop('Geography', axis=1, inplace=True)

# Label encoding
input_data['Gender'] = le.transform(input_data['Gender'])

# Scaling
input_data = scaler.transform(input_data)

# select the model for prediction
if service == 'ML Model':
    prediction = ml_model.predict(input_data)[0]
else:
    prediction = dl_model.predict(input_data)
    prediction = 1 if prediction[0][0] > 0.5 else 0

if prediction == 0:
    st.write('The customer will not churn')
else:
    st.write('The customer will churn')    