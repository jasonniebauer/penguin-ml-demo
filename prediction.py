import streamlit as st
import joblib


@st.cache_resource(show_spinner="Loading classifier model...")
def load_model():
    model = joblib.load('penguin_classifier_model.sav')
    return model

# Load once, reuse everywhere
MODEL = load_model()

def predict(data):
    return MODEL.predict(data)

def predict_probability(data):
    return MODEL.predict_proba(data)