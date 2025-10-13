import streamlit as st
import joblib
import pickle
import os

# ----------------------------
# Load Models (with Error Handling)
# ----------------------------
@st.cache_resource
def load_models():
    try:
        if os.path.exists("vectorizer.pkl") and os.path.exists("commit_classifier.pkl"):
            try:
                # Try loading with joblib first
                vectorizer = joblib.load("vectorizer.pkl")
                model = joblib.load("commit_classifier.pkl")
            except Exception:
                # If joblib fails, fall back to pickle
                with open("vectorizer.pkl", "rb") as f:
                    vectorizer = pickle.load(f)
                with open("commit_classifier.pkl", "rb") as f:
                    model = pickle.load(f)
            return vectorizer, model
        else:
            st.error("❌ Model files not found! Make sure 'vectorizer.pkl' and 'commit_classifier.pkl' are in the same folder.")
            st.stop()
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing module: {e.name}. Add it to requirements.txt.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.stop()

# Load model and vectorizer
vectorizer, model = load_models()

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Commit Message Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Commit Message Classifier")
st.write("This app predicts the category of a commit message using your trained ML model.")

st.markdown("---")

# Text input
user_input = st.text_area(
    "✍️ Enter a commit message:",
    height=150,
    placeholder="e.g., Fixed login bug in authentication module"
)

# Prediction
if st.button("🔍 Classify Commit"):
    if user_input.strip():
        try:
            # Transform input and predict
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]

            # Show prediction
            st.success(f"✅ **Predicted Category:** {prediction}")

            # Optional: show confidence (if model supports predict_proba)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_vec).max() * 100
                st.progress(int(prob))
                st.info(f"Confidence: {prob:.2f}%")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("⚠️ Please enter a commit message first.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Developed by [Your Name]")
