import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder

# Set the title of the app
st.title('Credit Card Default Prediction')

# File uploader for CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    creditcard_df = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write("Dataset:")
    st.dataframe(creditcard_df.head())

    # Display the description of the dataset
    st.write("Dataset Description:")
    st.write(creditcard_df.describe())

    # Check for missing values
    st.write("Missing Values Heatmap:")
    plt.figure(figsize=(10, 5))
    sns.heatmap(creditcard_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")
    st.pyplot(plt)

    # Drop the ID column
    if 'ID' in creditcard_df.columns:
        creditcard_df.drop(['ID'], axis=1, inplace=True)

    # Correlation heatmap
    st.write("Correlation Heatmap:")
    correlations = creditcard_df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlations, annot=True)
    st.pyplot(plt)

    # Age distribution
    st.write("Age Distribution by Default:")
    plt.figure(figsize=(12, 6))
    sns.countplot(x='AGE', hue='default.payment.next.month', data=creditcard_df)
    st.pyplot(plt)

    # One-hot encoding for categorical variables
    X_cat = creditcard_df[['SEX', 'EDUCATION', 'MARRIAGE']]
    onehotencoder = OneHotEncoder()
    X_cat = onehotencoder.fit_transform(X_cat).toarray()
    X_cat = pd.DataFrame(X_cat)

    # Select numerical features
    X_numerical = creditcard_df[['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                                  'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                                  'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
    
    # Combine categorical and numerical features
    X_all = pd.concat([X_cat, X_numerical], axis=1)
    X_all.columns = X_all.columns.astype(str)

    # Scale the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_all)
    y = creditcard_df['default.payment.next.month']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Set up and train an XGBoost classifier
    model = XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Initial Model Accuracy: {accuracy * 100:.2f} %")

    # Confusion Matrix
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt)

    # Classification Report
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Option for parameter tuning
    if st.checkbox('Tune model parameters'):
        st.write("Hyperparameter Tuning (This may take some time)...")

        param_grid = {
            'gamma': [0.5, 1, 5],   
            'subsample': [0.6, 0.8, 1.0], 
            'colsample_bytree': [0.6, 0.8, 1.0], 
            'max_depth': [3, 4, 5]
        }
        
        grid = GridSearchCV(model, param_grid, refit=True, verbose=4, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Predictions and evaluation
        y_predict_optim = grid.predict(X_test)
        accuracy_optim = accuracy_score(y_test, y_predict_optim)
        st.write(f"Optimized Model Accuracy: {accuracy_optim * 100:.2f}%")

        # Confusion Matrix after tuning
        cm_optim = confusion_matrix(y_test, y_predict_optim)
        st.subheader("Optimized Confusion Matrix")
        fig_optim, ax_optim = plt.subplots()
        sns.heatmap(cm_optim, annot=True, fmt='d', ax=ax_optim, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig_optim)

        # Best parameters from GridSearchCV
        st.write("Best Parameters from Tuning:")
        st.write(grid.best_params_)

# Run the app with the command: streamlit run app.py
