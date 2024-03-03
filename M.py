#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib


# In[11]:


# Step 1: Data Preprocessing
data = pd.read_csv('Task 3 and 4_Loan_Data.csv')


# In[12]:


# Feature Extraction
def extract_features(data):
    # Assuming features are already extracted from the dataset
    features = data[['credit_lines_outstanding', 'debt_to_income', 'payment_to_income', 'years_employed', 'fico_score']]
    return features


# In[13]:


# Calculate the payment_to_income ratio
data['payment_to_income'] = data['loan_amt_outstanding'] / data['income']
    
# Calculate the debt_to_income ratio
data['debt_to_income'] = data['total_debt_outstanding'] / data['income']


# In[14]:


X = extract_features(data)
y = data['default']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[17]:


# Step 2: Hyperparameter Tuning and Model Training
# Logistic Regression
logreg_params = {'C': [0.001, 0.01, 0.1, 1, 10]}
logreg_grid = GridSearchCV(LogisticRegression(), logreg_params, cv=5)
logreg_grid.fit(X_train_scaled, y_train)


# In[18]:


# Step 3: Model Evaluation
logreg_model = logreg_grid.best_estimator_
logreg_pred = logreg_model.predict(X_test_scaled)


# In[19]:


# Step 1: Load the trained Logistic Regression model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')


# In[20]:


# Step 2: Streamlit Deployment
st.title('Loan Default Prediction')


# In[21]:


def main():
    st.write('Enter customer details:')
    credit_lines_outstanding = st.number_input('Credit Lines Outstanding', min_value=0)
    debt_to_income = st.number_input('Debt to Income Ratio', min_value=0.00, format="%.4f")
    payment_to_income = st.number_input('Payment to Income Ratio', min_value=0.00, format="%.4f")
    years_employed = st.number_input('Years Employed', min_value=0)
    fico_score = st.number_input('FICO Score', min_value=0)

    if st.button('Predict'):
        # Transform input data
        input_data = pd.DataFrame({'credit_lines_outstanding': [credit_lines_outstanding],
                                   'debt_to_income': [debt_to_income],
                                   'payment_to_income': [payment_to_income],
                                   'years_employed': [years_employed],
                                   'fico_score': [fico_score]})
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Display prediction result
        if prediction[0] == 1:
            st.error('Risk of Default: Yes')
        else:
            st.success('Risk of Default: No')

if __name__ == '__main__':
    main()

