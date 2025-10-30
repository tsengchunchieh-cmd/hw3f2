import streamlit as st
import joblib
import re
import pandas as pd
import os

# Define the clean_text function (copied from train_baseline.py)
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Load the model and vectorizer
# Assuming artifacts are in the root directory for simplicity in this app.py
# In a real-world scenario, you might want to specify a more robust path.
try:
    model_path = os.path.join("artifacts", "model_svm.joblib")
    vectorizer_path = os.path.join("artifacts", "vectorizer.joblib")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please ensure 'model_svm.joblib' and 'vectorizer.joblib' are in the 'artifacts' directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

st.title("Spam Classifier")
st.write("Enter a message below to check if it's spam or not.")

# Text input from user
user_input = st.text_area("Message", "")

if st.button("Classify"):
    if user_input:
        # Preprocess the input
        cleaned_input = clean_text(user_input)
        
        # Transform the input using the loaded vectorizer
        input_vectorized = vectorizer.transform([cleaned_input])
        
        # Make prediction
        prediction = model.predict(input_vectorized)
        
        # Display result
        if prediction[0] == 1:
            st.error("This message is SPAM!")
        else:
            st.success("This message is NOT SPAM.")
    else:
        st.warning("Please enter a message to classify.")
