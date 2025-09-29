import streamlit as st
import pickle

# Load the model and vectorizer
@st.cache_resource
def load_model():
    with open("commit_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.set_page_config(page_title="Commit Classifier", page_icon="📊", layout="centered")

st.title("📊 Commit Message Classifier")
st.write("Enter a commit message below to predict its category.")

# Text input
user_input = st.text_area("Commit Message:", placeholder="Type a commit message here...")

if st.button("Classify"):
    if user_input.strip():
        # Transform input
        X = vectorizer.transform([user_input])
        # Predict
        prediction = model.predict(X)[0]

        # If model has probability support
        try:
            proba = model.predict_proba(X)[0]
            st.success(f"**Prediction:** {prediction}")
            st.write("### Class Probabilities")
            for cls, p in zip(model.classes_, proba):
                st.write(f"- {cls}: {p:.2f}")
        except Exception:
            st.success(f"**Prediction:** {prediction}")
    else:
        st.warning("⚠️ Please enter a commit message before classifying.")
```
