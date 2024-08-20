

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load the synthetic dataset
csv_file_path = '/content/heart_disease_prediction.csv'
data = pd.read_csv(csv_file_path)

# Prepare the dataset
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# Ensure 'Heart Disease' column is numeric
y = y.astype(int)

# Define categorical features
categorical_features = ['Sex']

# Convert categorical columns to string if they are not already
X[categorical_features] = X[categorical_features].astype(str)

# Check and handle missing values
X = X.dropna()
y = y[X.index]  # Align y with the cleaned X

# Create a column transformer to preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Define a function to predict heart disease
def predict_heart_disease(age, sex, cholesterol, blood_pressure):
    # Create DataFrame for new input
    new_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure]
    })
    new_data['Sex'] = new_data['Sex'].astype(str)

    # Predict using the fitted pipeline
    prediction = pipeline.predict(new_data)

    # Return the prediction result
    return "Heart Disease" if prediction[0] == 1 else "No Heart Disease"

# Streamlit app for heart disease prediction
def app():
    st.title('Heart Disease Prediction')
    
    age = st.number_input("Enter age:", min_value=20, max_value=80, step=1)
    sex = st.selectbox("Select sex:", [0, 1])  # 0 for female, 1 for male
    cholesterol = st.number_input("Enter cholesterol level:", min_value=150, max_value=300, step=1)
    blood_pressure = st.number_input("Enter blood pressure:", min_value=80, max_value=180, step=1)
    
    if st.button('Predict'):
        prediction = predict_heart_disease(age, sex, cholesterol, blood_pressure)
        st.write("Prediction Result:", prediction)

# Run the Streamlit app
if __name__ == "__main__":
    app()
