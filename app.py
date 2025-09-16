import streamlit as st
import re
import numpy as np
import pickle
from gensim.models import Word2Vec

@st.cache_resource
def load_models():
    rf_model = pickle.load(open("Sentiment_Analysis.pkl", "rb"))
    w2v_model = Word2Vec.load("word2vec.model")
    return rf_model, w2v_model

rf_model, w2v_model = load_models()

def get_vector(doc,model):
    #remove out-of-vocabulary words
    doc=[word for word in doc.split() if word in w2v_model.wv.index_to_key]
    return np.mean(w2v_model.wv[doc],axis=0)

def predict_sentiment(text):
    vec = get_vector(text, w2v_model).reshape(1, -1)
    pred = rf_model.predict(vec)[0]
    mapping = {0: "Negative 😞", 1: "Positive 😀"}
    return mapping.get(pred, "Unknown")

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("📊 Sentiment Analysis")
st.write("Enter your review below to analyze its sentiment.")

user_input = st.text_area("✍️ Enter your Review: ")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.subheader("Result:")
        st.success(sentiment)
    else:
        st.warning("⚠️ Please enter text.")

st.markdown("---")


