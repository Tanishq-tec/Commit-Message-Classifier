import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configure the page
st.set_page_config(
    page_title="Commit Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: #4ECDC4;
        color: white;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4ECDC4, #45B7D1);
    }
</style>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('commit_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    return vectorizer, classifier

# Header
st.markdown('<h1 class="main-header">🤖 Commit Classifier</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Control Panel")
    theme = st.selectbox("Theme", ["Default", "Minimal", "Modern"])
    if st.button("Stop"):
        st.error("System Halted!")
        st.stop()

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Commit Message")
    commit_message = st.text_area(
        "Commit message:",
        height=100,
        placeholder="feat: add authentication..."
    )

    if st.button("Analyze Commit", use_container_width=True):
        if commit_message:
            with st.spinner('Analyzing...'):
                vectorizer, classifier = load_models()
                features = vectorizer.transform([commit_message])
                prediction = classifier.predict(features)[0]
                probabilities = classifier.predict_proba(features)[0]

                class_names = classifier.classes_
                max_prob = max(probabilities)
                predicted_class = class_names[np.argmax(probabilities)]

                st.markdown(f"""
                <div class="prediction-box">
                    <h2>PREDICTION: {predicted_class.upper()}</h2>
                    <h3>Confidence: {max_prob*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)

                st.subheader("Confidence Breakdown")
                for class_name, prob in zip(class_names, probabilities):
                    st.progress(prob, text=f"{class_name}: {prob*100:.1f}%")

                st.subheader("Probability Radar")
                fig = go.Figure(data=go.Scatterpolar(
                    r=probabilities,
                    theta=class_names,
                    fill='toself',
                    line=dict(color='#4ECDC4')
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                fun_facts = {
                    'enhance file': "Your commit enhances the system!",
                    'fix issues': "Bug fixed successfully!",
                    'update docs': "Documentation updated!",
                    'refactor payment': "Payment system improved!",
                    'add email': "Email functionality added!",
                    'optimize database': "Database optimized!",
                    'implement jwt': "Security strengthened!",
                    'fix memory': "Memory issue resolved!",
                    'update api': "API upgraded!",
                    'add unit': "Unit tests added!"
                }

                # ✅ Fixed f-string issue
                st.info(f"💡 Fun Fact: {fun_facts.get(predicted_class, 'You are improving the project!')}")

        else:
            st.warning("Please enter a commit message.")

with col2:
    st.subheader("Quick Commit Ideas")
    sample_commits = [
        "fix: resolve memory leak in user service",
        "feat: add jwt authentication for users",
        "docs: update installation guide",
        "refactor: optimize database queries",
        "test: add unit tests for payment module"
    ]
    if st.button("Generate Random Commit"):
        st.text_area("Try this:", random.choice(sample_commits), height=80)

# Footer
st.markdown("---")
col6, col7 = st.columns(2)

with col6:
    st.write("### System Status")
    st.success("Neural Networks: Online")

with col7:
    st.write("### System Time")
    current_time = datetime.now().strftime("%H:%M:%S")
    st.write(f"⏱ {current_time}")
