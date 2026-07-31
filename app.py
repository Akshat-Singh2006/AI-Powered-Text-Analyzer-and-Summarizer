import streamlit as st
from transformers import pipeline
import pandas as pd

# Set up the page
st.set_page_config(page_title="AI Text Analyzer", page_icon="🤖")
st.title("🤖 AI Text Analyzer & Categorizer")
st.write("Paste text below to analyze sentiment, categorize it dynamically, and extract key entities.")

# Load models (cached so they don't reload on every button click)
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_ner():
    # aggregation_strategy="simple" groups sub-words into full names automatically
    return pipeline("ner", aggregation_strategy="simple")

sentiment_analyzer = load_sentiment_analyzer()
zero_shot_classifier = load_zero_shot()
ner_pipeline = load_ner()

# User Input
user_input = st.text_area("Enter your text here:", height=200)
categories = st.text_input("Enter custom categories separated by commas (e.g., Urgent, Spam, Inquiry):", "Urgent, Spam, Inquiry")

if st.button("Analyze Text"):
    if user_input and categories:
        with st.spinner("AI is crunching the data..."):
            # 1. Sentiment Analysis
            sentiment_result = sentiment_analyzer(user_input[:512])[0]
            
            # 2. Zero-Shot Classification
            candidate_labels = [label.strip() for label in categories.split(",")]
            zs_result = zero_shot_classifier(user_input, candidate_labels)
            
            # 3. Named Entity Recognition (NER)
            entities = ner_pipeline(user_input[:512])
            
            # --- DISPLAY RESULTS ---
            
            st.subheader("📊 Sentiment Analysis")
            st.write(f"**Tone:** {sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})")
            
            st.subheader("🏷️ Zero-Shot Categorization")
            st.write("Category distribution based on your custom labels:")
            # Create a dataframe for a clean bar chart
            chart_data = pd.DataFrame(
                {"Match Percentage": [score * 100 for score in zs_result['scores']]},
                index=zs_result['labels']
            )
            st.bar_chart(chart_data)

            st.subheader("🔍 Key Entities Extracted (NER)")
            if entities:
                st.write("The AI found the following proper nouns and entities:")
                for ent in entities:
                    st.write(f"- **{ent['word']}** (Type: {ent['entity_group']}, Confidence: {ent['score']:.2f})")
            else:
                st.write("No distinct people, locations, or organizations were found in this text.")
                
    else:
        st.warning("Please enter some text and categories to analyze!")