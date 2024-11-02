# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    return df

df = load_data()

# Streamlit app title with heart symbol
st.markdown("<h1 style='text-align: center; color: white;'>❤️ Heart Disease Prediction App </h1>", unsafe_allow_html=True)

# Add a colorful sidebar
st.sidebar.markdown("<h2 style='color: teal;'>Input Features</h2>", unsafe_allow_html=True)

# Show the dataset if the user wants to view it
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Data Preprocessing
X = df.drop('target', axis=1)  # Features (everything except 'target')
y = df['target']               # Target (heart disease indicator)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_rf_model = grid_search.best_estimator_

# Sidebar for user input parameters
def user_input_features():
    features = {}
    for col in X.columns:
        features[col] = st.sidebar.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    return pd.DataFrame([features])

input_df = user_input_features()

# Display the user's input features
st.subheader("User Input Parameters")
st.write(input_df)

# Make predictions based on user input
prediction = best_rf_model.predict(input_df)
prediction_proba = best_rf_model.predict_proba(input_df)

# Get prediction probability for the "Heart Disease" class
heart_disease_probability = prediction_proba[0][1]

# Determine the intensity of heart disease
if heart_disease_probability < 0.4:
    intensity = "Low risk of heart disease."
elif 0.4 <= heart_disease_probability < 0.7:
    intensity = "Moderate risk of heart disease."
else:
    intensity = "High risk of heart disease."

# Display predictions
st.subheader("Prediction")
heart_disease_label = np.array(['No Heart Disease', 'Heart Disease'])
st.write(f"<h3 style='color: blue;'>{heart_disease_label[prediction][0]}</h3>", unsafe_allow_html=True)

# Show intensity of heart disease instead of probabilities
st.subheader("Intensity of Heart Disease")
st.write(f"<h4 style='color: orange;'>{intensity}</h4>", unsafe_allow_html=True)

# Evaluate the model on the test data
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Show the accuracy, precision, and recall
st.subheader("Model Evaluation Metrics")
st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
st.write(f"**Precision:** {precision * 100:.2f}%")
st.write(f"**Recall:** {recall * 100:.2f}%")

# Custom styling for the app with a gradient background and heart symbol
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #80deea;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a heart symbol at the bottom of the app
st.markdown("<h4 style='text-align: center; color: white;'>Thank you for using our app!</h4>", unsafe_allow_html=True)
