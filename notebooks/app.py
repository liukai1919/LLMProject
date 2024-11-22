import streamlit as st
from transformers import pipeline

# Initialize the Streamlit app
st.title("Sentiment Analysis App")
st.write("Analyze the sentiment of your text using a Hugging Face Transformers model.")

# Text input from the user
user_input = st.text_area("Enter text for sentiment analysis:", "")

# Load the sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="liukai1987/distilbert-base-uncased-imdb_finetune", device='mps')

model = load_model()

# Perform sentiment analysis if the user enters text
if user_input:
    with st.spinner("Analyzing sentiment..."):
        result = model(user_input)
        sentiment = result[0]['label']
        score = result[0]['score']

    # Display the results
    st.write("**Analysis Result:**")
    if sentiment == "POSITIVE":
        st.success(f"Positive sentiment (confidence: {score:.2f})")
    else:
        st.error(f"Negative sentiment (confidence: {score:.2f})")
