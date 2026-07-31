import streamlit as st
from transformers import pipeline
import pandas as pd
import math

# Set up the page
st.set_page_config(page_title="AI Text Analyzer", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise AI Text Analyzer")
st.write("Analyze sentiment, extract entities, categorize dynamically, and export your data.")

# Load models 
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_ner():
    return pipeline("ner", aggregation_strategy="simple")

sentiment_analyzer = load_sentiment_analyzer()
zero_shot_classifier = load_zero_shot()
ner_pipeline = load_ner()

# User Input
user_input = st.text_area("Enter your text here:", height=150)
categories = st.text_input("Enter custom categories separated by commas:", "Urgent, Spam, Inquiry")

if st.button("Run AI Analysis"):
    if user_input and categories:
        with st.spinner("AI is crunching the data..."):
            
            # --- TEXT ANALYTICS ---
            word_count = len(user_input.split())
            reading_time = math.ceil(word_count / 200) # Avg reading speed
            
            # --- AI PROCESSING ---
            sentiment_result = sentiment_analyzer(user_input[:512])[0]
            candidate_labels = [label.strip() for label in categories.split(",")]
            zs_result = zero_shot_classifier(user_input, candidate_labels)
            entities = ner_pipeline(user_input[:512])
            
            # --- UI LAYOUT ---
            st.divider()
            
            # Top Row: Stats & Sentiment
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Text Statistics & Sentiment")
                st.write(f"**Word Count:** {word_count} words")
                st.write(f"**Est. Reading Time:** {reading_time} min")
                st.write(f"**Tone:** {sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})")
                
            with col2:
                st.subheader("🔍 Key Entities Extracted (NER)")
                if entities:
                    for ent in entities:
                        st.write(f"- **{ent['word']}** ({ent['entity_group']})")
                else:
                    st.write("No distinct people, locations, or organizations found.")

            st.divider()
            
            # Bottom Row: Zero Shot & Export
            st.subheader("🏷️ Dynamic Categorization")
            
            chart_data = pd.DataFrame(
                {"Match Percentage": [score * 100 for score in zs_result['scores']]},
                index=zs_result['labels']
            )
            st.bar_chart(chart_data)
            
            # CSV EXPORT FEATURE
            csv = chart_data.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Categorization Data as CSV",
                data=csv,
                file_name='ai_categorization_results.csv',
                mime='text/csv',
            )
                
    else:
        st.warning("Please enter some text and categories to analyze!")