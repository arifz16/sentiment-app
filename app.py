# app.py
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# Load model dan vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing sama seperti di training
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# UI Streamlit
st.title("Analisis Sentimen Review Film")
review_input = st.text_area("Masukkan review film:")

if st.button("Prediksi"):
    if review_input.strip() == "":
        st.warning("Masukkan review terlebih dahulu.")
    else:
        clean_text = preprocess(review_input)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)
        label = "Positif" if prediction[0] == 1 else "Negatif"
        st.success(f"Hasil Prediksi: **{label}**")
