import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Page Config
st.set_page_config(page_title="MindGuard AI", page_icon="🧠")

# 2. Model Training (Cached for speed)
@st.cache_resource
def train_model():
    data = {
        'text': [
            "I'm feeling wonderful", "Best day ever", "I am happy", "Life is good",
            "I feel so empty and sad", "I can't stop crying", "I want to disappear",
            "Just a normal day", "The weather is okay", "I'm going to the gym",
            "Everything is hopeless", "I hate being alone", "I'm so tired of this",
            "I love my friends", "Had a great lunch", "I feel worthless today"
        ] * 10,
        'label': [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1] * 10
    }
    df = pd.DataFrame(data)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('lr', LogisticRegression())
    ])
    pipeline.fit(df['text'], df['label'])
    return pipeline

model = train_model()

# 3. UI
st.title("🧠 MindGuard Sentiment Analytics")
st.write("Detecting emotional markers in text using Machine Learning.")

user_input = st.text_area("How are you feeling?", "I've been feeling a bit low lately.")

if st.button("Analyze"):
    probs = model.predict_proba([user_input])[0]
    prediction = model.predict([user_input])[0]
    
    st.divider() # This is the correct way to add a line!
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("### Potential Distress Detected")
        else:
            st.success("### Sentiment Appears Stable")
            
    with col2:
        prob_df = pd.DataFrame({
            'Sentiment': ['Stable', 'Distress'],
            'Confidence': [probs[0], probs[1]]
        })
        fig = px.bar(prob_df, x='Sentiment', y='Confidence', color='Sentiment',
                     color_discrete_map={'Stable': '#2ecc71', 'Distress': '#e74c3c'})
        fig.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
