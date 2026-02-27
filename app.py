import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Create a tiny dataset (50 samples simplified)
data = {
    'text': [
        "I am so happy today", "The weather is lovely", "I love my life", "Feeling great",
        "I feel empty inside", "I can't get out of bed", "I hate myself", "Everything is dark",
        "Going for a walk", "Enjoying this coffee", "Life is beautiful", "I feel hopeless",
        "I want to give up", "I am tired of crying", "Just finished a book", "Feeling lonely",
        "The sun is shining", "I am so lonely and sad", "I feel worthless", "Great workout!"
    ] * 5, # Multiplying to get 100 samples quickly
    'label': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0] * 5
}
df = pd.DataFrame(data)

# 2. Preprocessing & Model Training
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X = vectorizer.fit_transform(df['text'])
model = LogisticRegression()
model.fit(X, df['label'])

# 3. Streamlit Web App Interface
st.title("🧠 Mental Health Sentiment Check")
st.write("Enter text below to see if it shows signs of depression/anxiety.")

user_input = st.text_input("How are you feeling?", "I've been feeling very lonely lately.")

if st.button("Analyze"):
    # Transform input and predict
    processed_input = vectorizer.transform([user_input])
    prediction = model.predict(processed_input)
    
    if prediction[0] == 1:
        st.error("Result: Potential Signs of Distress Detected")
        st.write("Please reach out to a professional if you are struggling.")
    else:
        st.success("Result: Normal/Positive Sentiment Detected")
