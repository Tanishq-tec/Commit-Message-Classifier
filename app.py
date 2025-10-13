import streamlit as st
import pickle

# ----------------------------
# Load the trained model and vectorizer
# ----------------------------
@st.cache_resource
def load_models():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("commit_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_models()

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Commit Message Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Commit Message Classifier")
st.write("This app predicts the type of a commit message using a trained ML model.")

# Input box for commit message
user_input = st.text_area("Enter a commit message:", height=150, placeholder="e.g. Fixed a bug in the login function")

# Predict button
if st.button("Classify"):
    if user_input.strip():
        # Transform the input text
        input_vec = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_vec)[0]
        
        st.success(f"✅ Predicted Category: **{prediction}**")
    else:
        st.warning("⚠️ Please enter a commit message to classify.")

# Optional Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")

