import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- 1. CONFIG & DATA ---
st.set_page_config(page_title="MindGuard AI", page_icon="🧠")

@st.cache_resource # Keeps the model in memory so it doesn't retrain on every click
def train_model():
    # Expanded synthetic dataset for better variety
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
    
    # Using a Pipeline to bundle preprocessing and modeling
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('lr', LogisticRegression())
    ])
    
    pipeline.fit(df['text'], df['label'])
    return pipeline

model = train_model()

# --- 2. UI INTERFACE ---
st.title("🧠 MindGuard Sentiment Analytics")
st.markdown("""
    This tool uses **Natural Language Processing (NLP)** to detect emotional distress.
    *Note: This is an AI demo, not a clinical diagnosis.*
""")

with st.container():
    user_input = st.text_area("Share what's on your mind...", placeholder="e.g., I've been feeling quite overwhelmed lately.")
    analyze_btn = st.button("Analyze Sentiment", use_container_width=True)

---

# --- 3. LOGIC & VISUALIZATION ---
if analyze_btn and user_input:
    # Get probability scores
    probs = model.predict_proba([user_input])[0]
    prediction = model.predict([user_input])[0]
    
    col1, col2 = st.columns([1, 1])

    with col1:
        if prediction == 1:
            st.error("### Result: Distress Detected")
            st.write("The model identified markers often associated with depression or anxiety.")
        else:
            st.success("### Result: Stable/Positive")
            st.write("The model identified a generally positive or neutral tone.")

    with col2:
        # Visualizing the confidence levels
        prob_df = pd.DataFrame({
            'Sentiment': ['Stable', 'Distress'],
            'Confidence': [probs[0], probs[1]]
        })
        fig = px.bar(prob_df, x='Sentiment', y='Confidence', color='Sentiment',
                     color_discrete_map={'Stable': '#2ecc71', 'Distress': '#e74c3c'})
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Resource Section
    if prediction == 1:
        st.divider()
        st.warning("💛 **You're not alone.** If you're struggling, consider reaching out to a professional or a crisis hotline (e.g., 988 in the US).")
