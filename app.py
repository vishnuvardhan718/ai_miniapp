import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="MindGuard AI",
    page_icon="🧠",
    layout="centered"
)

# 2. DATA & MODEL TRAINING
# We use @st.cache_resource so the model only trains ONCE when the app starts
@st.cache_resource
def load_trained_model():
    # Synthetic dataset for demonstration
    data = {
        'text': [
            "I am so happy", "Life is great", "I love everything", "Feeling wonderful",
            "I feel so empty", "I can't stop crying", "I hate myself", "I want to give up",
            "Going for a walk", "Drinking some tea", "Just a normal day", "The sky is blue",
            "Everything is dark", "I feel hopeless", "I am so lonely", "I'm exhausted"
        ] * 20, # Expanded to 320 samples for better stability
        'label': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1] * 20
    }
    df = pd.DataFrame(data)
    
    # Pipeline bundles the Vectorizer and the Model together
    # This prevents errors during text transformation
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('lr', LogisticRegression())
    ])
    
    pipeline.fit(df['text'], df['label'])
    return pipeline

# Load the model into memory
model = load_trained_model()

# 3. USER INTERFACE
st.title("🧠 MindGuard: Sentiment Analysis")
st.markdown("""
    Enter text below to analyze emotional markers. This tool uses a 
    **Logistic Regression** model with **TF-IDF Vectorization**.
""")

# Input Area
user_input = st.text_area(
    "How are you feeling?", 
    placeholder="Type your thoughts here...",
    help="The AI will look for patterns often associated with high-stress or low-mood sentiments."
)

# Analysis Logic
if st.button("Run Analysis", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Get Predictions
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]
        
        # UI Visual Separation
        st.divider()
        
        # Results Columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if prediction == 1:
                st.error("### Result: Potential Distress")
                st.info("The model detected markers of sadness or anxiety. Please consider talking to someone you trust.")
            else:
                st.success("### Result: Generally Stable")
                st.info("The model detected a neutral or positive emotional tone.")
        
        with col2:
            # Bar Chart showing confidence
            # label 0 = Stable, label 1 = Distress
            prob_df = pd.DataFrame({
                'Category': ['Stable', 'Distress'],
                'Confidence (%)': [probabilities[0] * 100, probabilities[1] * 100]
            })
            
            fig = px.bar(
                prob_df, 
                x='Category', 
                y='Confidence (%)',
                color='Category',
                color_discrete_map={'Stable': '#2ecc71', 'Distress': '#e74c3c'},
                text_auto='.1f'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# 4. FOOTER
st.divider()
st.caption("Disclaimer: This is an AI demonstration and does not provide medical advice or diagnosis.")
