import streamlit as st
st.set_page_config(page_title="🚨 Racism Detection", layout="centered")

import numpy as np
import pickle
import json
import re
import os
import nltk

# Ensure NLTK resources are downloaded
nltk_data_dir = os.path.expanduser("~/nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Download required NLTK datasets if not already present
nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources if not already done
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# Streamlit UI
st.title("🚨 Racism Detection in Tweets")
st.markdown("Enter a tweet below to check if it contains racist language.")

tweet_input = st.text_area("Tweet Input:", height=150, placeholder="Type or paste a tweet here...")

# Load model, tokenizer, and config
@st.cache(allow_output_mutation=True)
def load_assets():
    model = load_model("my_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("config.json", "r") as f:
        config = json.load(f)
    return model, tokenizer, config["max_len"]

model, tokenizer, MAX_LEN = load_assets()

# Text Cleaning Pipeline
def text_cleaning_pipeline(text, rule="lemmatize"):
    def lower_order(t): return t.lower()
    def remove_urls(t): return re.sub(r'https?://\S+|www\.\S+', '', t)
    def remove_emoji(string):
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub(r'', string)
    def remove_unwanted_characters(document):
        document = re.sub(r'@[A-Za-z0-9_]+', '', document)
        document = re.sub(r'#[A-Za-z0-9_]+', '', document)
        document = re.sub(r'[^\w\s]', '', document)
        return document.strip()

    # Tokenization
    def tokenize(text):
        return nltk.word_tokenize(text)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    def remove_stopwords(tokens):
        return [token for token in tokens if token not in stop_words]

    # Lemmatization or Stemming
    def lemmatize(tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    def stem(tokens):
        stemmer = PorterStemmer()
        return [stemmer.stem(token) for token in tokens]

    # Apply cleaning steps
    text = lower_order(text)
    text = remove_urls(text)
    text = remove_emoji(text)
    text = remove_unwanted_characters(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if rule == "lemmatize":
        tokens = lemmatize(tokens)
    elif rule == "stem":
        tokens = stem(tokens)
    return " ".join(tokens)

# Prediction Function
def predict_tweet(tweet):
    cleaned_text = text_cleaning_pipeline(tweet)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded_seq)[0][0]
    label = "Racist" if prediction > 0.5 else "Not Racist"
    return label, prediction

if st.button("Predict"):
    if not tweet_input.strip():
        st.warning("Please enter a tweet.")
    else:
        label, score = predict_tweet(tweet_input)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence Score: **{score:.4f}**")
        st.progress(score if label == "Racist" else 1 - score)