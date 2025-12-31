# app.py

import streamlit as st
import joblib

# Load model
model = joblib.load("saved_model/sentiment_model.pkl")

# App UI
st.title("🎬 Sentiment Analysis on Movie Reviews")
st.write("Type a movie review and the model predicts sentiment!")

review = st.text_area("Enter your review text:")

if st.button("Predict"):
    if review:
        result = model.predict([review])
        st.write(f"**Predicted Sentiment:** {result[0]}")
    else:
        st.write("Please enter some text first.")
