# ------------------- Cấu hình ban đầu -------------------
# cd Downloads\Code
# venv\Scripts\activate
# pip install scikit-learn
# pip install streamlit transformers torch google-generativeai joblib
# streamlit run app.py

import streamlit as st
import pandas as pd
import torch
import time, random, joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai

st.set_page_config(page_title="Disaster Tweet Classifier", layout="centered")
st.title("🌪️ Disaster Tweet Multi-Task Classifier")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
genai.configure(api_key="AIzaSyDQinFxEPdmlucUSrepU7XBbKxcXZbcSgw")

@st.cache_resource
def load_models():
    model_task1 = AutoModelForSequenceClassification.from_pretrained("bertweet-disaster-task1").to(device).eval()
    tokenizer_task1 = AutoTokenizer.from_pretrained("bertweet-disaster-task1")

    model_task2 = AutoModelForSequenceClassification.from_pretrained("bertweet-disaster-task2").to(device).eval()
    tokenizer_task2 = AutoTokenizer.from_pretrained("bertweet-disaster-task2")

    le = joblib.load("label_encoder_task2.pkl")

    gemini_model = genai.GenerativeModel("models/gemini-2.0-flash-thinking-exp")

    return model_task1, tokenizer_task1, model_task2, tokenizer_task2, le, gemini_model

model_task1, tokenizer_task1, model_task2, tokenizer_task2, le, gemini_model = load_models()

def predict_task1(text):
    inputs = tokenizer_task1(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_task1(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

def predict_task2(text):
    inputs = tokenizer_task2(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_task2(**inputs)
    pred_index = torch.argmax(outputs.logits, dim=1).item()
    return le.inverse_transform([pred_index])[0]

def create_prompt(tweet_text):
    return f"""
    You are an AI assistant trained in meteorology and disaster event analysis. Your task is to classify tweets as belonging to: disaster_info, emergency_help, emotion_sharing.
    Return exactly:
    - disaster_info=1 or 0
    - emergency_help=1 or 0
    - emotion_sharing=1 or 0

    Tweet:
    ```
    {tweet_text}
    ```
    """

def predict_task3(text):
    prompt = create_prompt(text)
    try:
        time.sleep(random.uniform(2, 4))
        response = gemini_model.generate_content(prompt)
        output = response.text.strip()

        return {
            'p_disaster_info': 1 if "disaster_info=1" in output else 0,
            'p_emergency_help': 1 if "emergency_help=1" in output else 0,
            'p_emotion_sharing': 1 if "emotion_sharing=1" in output else 0
        }
    except Exception as e:
        st.warning("Gemini API error: " + str(e))
        return {'p_disaster_info': -1, 'p_emergency_help': -1, 'p_emotion_sharing': -1}

def run_pipeline(text):
    result = {}

    pred_task1 = predict_task1(text)
    result['Disaster Detection'] = "Disaster" if pred_task1 == 1 else "Non-disaster"

    result['Disaster Type'] = predict_task2(text) if pred_task1 == 1 else "non-disaster"

    task3 = predict_task3(text)
    if task3['p_disaster_info'] == -1:
        result['Intention Sentence'] = "Gemini API Error"
    else:
        labels = []
        if task3['p_disaster_info']: labels.append("disaster_info")
        if task3['p_emergency_help']: labels.append("emergency_help")
        if task3['p_emotion_sharing']: labels.append("emotion_sharing")
        result['Intention Sentence'] = ", ".join(labels) if labels else "None"

    return result

tweet_input = st.text_area("✍️ Enter a Tweet", height=150)

if st.button("🔍 Classify"):
    if tweet_input.strip() == "":
        st.warning("Please enter a tweet to classify.")
    else:
        with st.spinner("🔍 Predicting..."):
            prediction = run_pipeline(tweet_input)

        st.success("✅ Prediction Complete!")

        st.markdown("### 📌 **Results**")
        st.markdown(f"**Tweet:** {tweet_input}")
        st.markdown(f"**Disaster Detection:** {prediction['Disaster Detection']}")
        st.markdown(f"**Disaster Type:** {prediction['Disaster Type']}")
        st.markdown(f"**Intention Sentence:** {prediction['Intention Sentence']}")
