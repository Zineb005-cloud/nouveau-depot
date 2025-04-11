import streamlit as st
import joblib

st.title("Détection de sentiment texte")

model = joblib.load("model_sentiment.pkl")

user_input = st.text_input("Entrez une phrase")

if user_input:
    prediction = model.predict([user_input])
    if prediction[0] == 0:
        st.write("Sentiment détecté : Sport (négatif)")
    else:
        st.write("Sentiment détecté : Espace (positif)")
