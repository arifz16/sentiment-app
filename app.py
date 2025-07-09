# app.py
import streamlit as st
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns

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
# Sidebar menu
menu = st.sidebar.selectbox("Pilih Mode:", ["Review Manual", "Upload CSV"])

# Session state untuk riwayat
if "history" not in st.session_state:
    st.session_state.history = []

# =============== Review Manual ===============
if menu == "Review Manual":
    review_input = st.text_area("Masukkan review film:")

    if st.button("Prediksi Sentimen"):
        if review_input.strip() == "":
            st.warning("Masukkan review terlebih dahulu.")
        else:
            clean_text = preprocess(review_input)
            vector = vectorizer.transform([clean_text])
            prediction = model.predict(vector)[0]
            probas = model.predict_proba(vector)[0]
            label = "Positif" if prediction == 1 else "Negatif"

            st.success(f"Hasil Prediksi: **{label}**")
            st.write(f"Confidence Positif: {probas[1]:.2%}, Negatif: {probas[0]:.2%}")

            # Simpan riwayat
            st.session_state.history.append((review_input, label, probas[1]))

    # Tampilkan riwayat
    if st.session_state.history:
        st.subheader("Riwayat Prediksi")
        hist_df = pd.DataFrame(st.session_state.history, columns=["Review", "Sentimen", "Confidence Positif"])
        st.dataframe(hist_df)

# =============== Upload CSV ===============
if menu == "Upload CSV":
    uploaded_file = st.file_uploader("Upload file CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'review' not in df.columns:
            st.error("File harus memiliki kolom 'review'.")
        else:
            df['clean'] = df['review'].apply(preprocess)
            X = vectorizer.transform(df['clean'])
            df['sentiment'] = model.predict(X)
            df['confidence'] = model.predict_proba(X)[:, 1]
            df['sentiment'] = df['sentiment'].map({1: 'Positif', 0: 'Negatif'})

            st.subheader("Hasil Klasifikasi")
            st.dataframe(df[['review', 'sentiment', 'confidence']])

            # Visualisasi
            st.subheader("Distribusi Sentimen")
            fig, ax = plt.subplots()
            sns.countplot(x='sentiment', data=df, ax=ax)
            st.pyplot(fig)
