import streamlit as st
import pickle

# ----------------------------
# Load Models with Error Handling
# ----------------------------
@st.cache_resource
def load_models():
    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("commit_classifier.pkl", "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except ModuleNotFoundError as e:
        st.error(f"❌ Missing Python module: {e.name}. Please add it to requirements.txt.")
        raise e
    except FileNotFoundError as e:
        st.error("❌ Could not find one or more model files (vectorizer.pkl or commit_classifier.pkl).")
        raise e
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        raise e

# Load models
vectorizer, model = load_models()

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Commit Message Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Commit Message Classifier")
st.write("This app predicts the category of a commit message using a trained ML model.")

# Input box for commit message
user_input = st.text_area("✍️ Enter a commit message:", height=150, placeholder="e.g. Fixed login bug in authentication module")

# Predict button
if st.button("🔍 Classify"):
    if user_input.strip():
        try:
            # Transform input text
            input_vec = vectorizer.transform([user_input])

            # Predict label
            prediction = model.predict(input_vec)[0]

            # Display result
            st.success(f"✅ Predicted Category: **{prediction}**")

        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("⚠️ Please enter a commit message first.")

# Optional Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Developed by [Your Name]")
