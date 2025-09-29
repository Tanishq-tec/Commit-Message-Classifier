import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configure the page with a crazy theme
st.set_page_config(
    page_title="🚀 Commit Classifier 9000",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for maximum craziness
st.markdown("""
<style>
    .main-header {
        font-size: 4rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 900;
        margin-bottom: 2rem;
        animation: rainbow 2s ease-in-out infinite;
    }
    @keyframes rainbow {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00, #00ffff, #0000ff, #ff00ff);
    }
</style>
""", unsafe_allow_html=True)

# Load models with progress animation
@st.cache_resource
def load_models():
    with st.spinner('🚀 Activating neural networks...'):
        progress_bar = st.progress(0)
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('commit_classifier.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
        st.balloons()
        return vectorizer, classifier

# Crazy header
st.markdown('<h1 class="main-header">🤖 COMMIT CLASSIFIER 9000</h1>', unsafe_allow_html=True)

# Sidebar with crazy options
with st.sidebar:
    st.title("⚡ Control Panel")
    
    # Theme selector
    theme = st.selectbox(
        "🎨 Choose Your Vibe:",
        ["Cyberpunk", "Neon Dreams", "Matrix", "Retro Wave", "Space Odyssey"]
    )
    
    # Animation intensity
    intensity = st.slider("💥 Animation Intensity", 1, 10, 5)
    
    # Sound effects
    sound_effects = st.checkbox("🔊 Enable Sound Effects", value=True)
    
    # Secret mode
    if st.checkbox("🕵️‍♂️ Activate Secret Agent Mode"):
        st.success("Mission accepted! Classifying with extra stealth...")
    
    # Emergency button
    if st.button("🚨 EMERGENCY STOP"):
        st.error("SYSTEM HALTED!")
        st.stop()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Enter Your Commit Message")
    
    # Text input with style
    commit_message = st.text_area(
        "Type your commit message here:",
        height=100,
        placeholder="feat: add quantum encryption module for enhanced security...",
        help="Make it descriptive! The AI loves good commit messages! 🤓"
    )
    
    # Prediction button with flair
    if st.button("🧠 ANALYZE COMMIT", use_container_width=True):
        if commit_message:
            with st.spinner('🔮 Consulting the AI oracle...'):
                # Add some dramatic delay
                time.sleep(1.5)
                
                # Load models and predict
                vectorizer, classifier = load_models()
                features = vectorizer.transform([commit_message])
                prediction = classifier.predict(features)[0]
                probabilities = classifier.predict_proba(features)[0]
                
                # Get class names
                class_names = classifier.classes_
                
                # Create prediction result
                max_prob = max(probabilities)
                predicted_class = class_names[np.argmax(probabilities)]
                
                # Display results with crazy animations
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎉 PREDICTION: {predicted_class.upper()}</h2>
                    <h3>Confidence: {max_prob*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bars for all classes
                st.subheader("📊 Confidence Breakdown")
                for class_name, prob in zip(class_names, probabilities):
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.write(f"{class_name}:")
                    with col_b:
                        st.progress(prob, text=f"{prob*100:.1f}%")
                
                # Crazy visualization
                st.subheader("📈 Probability Radar")
                fig = go.Figure(data=go.Scatterpolar(
                    r=probabilities,
                    theta=class_names,
                    fill='toself',
                    line=dict(color='#FF6B6B')
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Fun fact based on prediction
                fun_facts = {
                    'enhance file': "✨ Your commit is enhancing the matrix!",
                    'fix issues': "🐛 Squashing bugs like a pro!",
                    'update docs': "📚 Documentation is love, documentation is life!",
                    'refactor payment': "💳 Making money moves!",
                    'add email': "📧 Email me maybe?",
                    'optimize database': "⚡ Speedy Gonzalez mode activated!",
                    'implement jwt': "🔐 Security level: Fort Knox!",
                    'fix memory': "🧠 Brain gains!",
                    'update api': "🌐 Connecting the dots!",
                    'add unit': "🧪 Science, bitch!"
                }
                
                st.info(f"💡 **Fun Fact**: {fun_facts.get(predicted_class, 'You're making the world a better place, one commit at a time!')}")
                
        else:
            st.warning("⚠️ Please enter a commit message to analyze!")

with col2:
    st.subheader("🎲 Quick Commit Ideas")
    
    # Sample commit messages
    sample_commits = [
        "fix: resolve memory leak in user service",
        "feat: add jwt authentication for users",
        "docs: update installation guide",
        "refactor: optimize database queries",
        "test: add unit tests for payment module"
    ]
    
    if st.button("🎪 Random Commit Generator"):
        random_commit = random.choice(sample_commits)
        st.text_area("Try this one:", random_commit, height=80)
    
    # Stats section
    st.subheader("📈 Today's Stats")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Commits Analyzed", "1,337", "+42")
    
    with col4:
        st.metric("Accuracy", "98.7%", "+0.2%")
    
    with col5:
        st.metric("AI Awesomeness", "∞", "MAX")
    
    # Leaderboard
    st.subheader("🏆 Top Categories")
    categories = ['fix issues', 'enhance file', 'update docs', 'refactor payment', 'add email']
    counts = [45, 32, 28, 19, 15]
    
    fig = px.bar(
        x=counts, 
        y=categories, 
        orientation='h',
        color=counts,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Footer with crazy elements
st.markdown("---")
col6, col7, col8 = st.columns(3)

with col6:
    st.write("### 🤖 AI Status")
    st.success("Neural Networks: ONLINE")
    st.info("Quantum Processor: ACTIVATED")
    st.warning("Caffeine Levels: CRITICAL")

with col7:
    st.write("### 🎯 Mission")
    st.write("Classifying commits with the power of **MACHINE LEARNING** and **PURE AWESOMENESS**!")
    
    # Live clock
    current_time = datetime.now().strftime("%H:%M:%S")
    st.write(f"🕐 System Time: {current_time}")

with col8:
    st.write("### 🚀 Next Level")
    if st.button("ACTIVATE TURBO MODE"):
        st.balloons()
        st.snow()
        st.success("TURBO MODE ACTIVATED! Preparing for hyperspace...")
        time.sleep(2)
        st.rerun()

# Secret Easter egg
if st.sidebar.checkbox("👑 Enable Royal Mode"):
    st.sidebar.success("Your majesty! The AI bows before you! 👑")
    st.balloons()