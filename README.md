# Credit Risk Analysis: Loan Default Prediction

## Introduction

Welcome to the Credit Risk Analysis project repository! This project focuses on predicting the likelihood of loan defaults using machine learning techniques. By analyzing historical data on customer attributes and loan outcomes, we aim to develop predictive models that assist financial institutions in making informed lending decisions and managing credit risk effectively.

## Analysis Overview

In this analysis, we covered the following key aspects:

- **Data Preprocessing:** Cleaned, extracted features, and transformed the dataset for model training. This involved handling missing values, scaling numerical features, and encoding categorical variables.
  
- **Exploratory Data Analysis (EDA):** Analyzed feature distribution, identified patterns, and detected outliers to gain insights into the data.

- **Model Training and Evaluation:** Utilized machine learning models such as Logistic Regression, Decision Tree, and Random Forest for loan default prediction. Employed hyperparameter tuning techniques using GridSearchCV to optimize model performance.

- **Feature Engineering:** Derived additional features such as payment to income ratio and debt to income ratio to enhance model predictive power.

- **Model Selection:** Evaluated the performance of each model based on metrics such as accuracy, precision, recall, and AUC-ROC score. Selected Logistic Regression as the best-performing model due to its superior performance across all evaluation metrics. Logistic Regression offers a good balance between model complexity and predictive accuracy, making it suitable for practical deployment in real-world scenarios.

- **Model Deployment:** Deployed the selected Logistic Regression model using Streamlit, enabling users to interact with the model via a web application.

## Model Selection

- **Logistic Regression:** Selected as the best-performing model due to its superior performance across all evaluation metrics. Achieved the highest accuracy, precision, recall, and AUC-ROC score among the three models evaluated.

## Streamlit Deployment

A Streamlit web application was developed to deploy the selected Logistic Regression model. Users can input borrower information and receive predictions on the likelihood of loan defaults.

## Conclusion

The Credit Risk Analysis project provides valuable insights into predicting loan defaults using machine learning techniques. By leveraging models such as Logistic Regression and deploying them via Streamlit, financial institutions can make more informed lending decisions and manage credit risk effectively.
