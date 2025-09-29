```python
import streamlit as st
import pandas as pd

# Try to import joblib safely
try:
    import joblib
except ImportError:
    st.error("❌ Missing dependency: joblib is not installed. Please add it to requirements.txt")
    st.stop()

# ----------- Load Model and Vectorizer -----------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("commit_classifier.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading model/vectorizer: {e}")
        st.stop()

model, vectorizer = load_model()

# ----------- Streamlit Page Config -----------
st.set_page_config(page_title="Commit Classifier", page_icon="📊", layout="centered")

st.title("📊 Commit Message Classifier")
st.write("This app predicts the category of commit messages using a trained ML model.")

# ----------- Tabs for Single / Batch Prediction -----------
tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch Prediction (CSV)"])

# ----------- Single Prediction -----------
with tab1:
    st.subheader("🔹 Classify a Single Commit Message")
    user_input = st.text_area("Commit Message:", placeholder="Type a commit message here...")

    if st.button("Classify", key="single"):
        if user_input.strip():
            try:
                X = vectorizer.transform([user_input])
                prediction = model.predict(X)[0]

                # Show probabilities if supported
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    st.success(f"**Prediction:** {prediction}")
                    st.write("### Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Class": model.classes_,
                        "Probability": proba
                    }).sort_values("Probability", ascending=False)
                    st.dataframe(prob_df, use_container_width=True)
                else:
                    st.success(f"**Prediction:** {prediction}")

            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
        else:
            st.warning("⚠️ Please enter a commit message before classifying.")

# ----------- Batch Prediction (CSV Upload) -----------
with tab2:
    st.subheader("📂 Upload a CSV file for Batch Prediction")
    st.write("The CSV should contain a column named `commit_message` with commit texts.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "commit_message" not in df.columns:
                st.error("❌ CSV must contain a column named `commit_message`.")
            else:
                X_batch = vectorizer.transform(df["commit_message"])
                preds = model.predict(X_batch)

                df["Prediction"] = preds

                # Add probabilities if available
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_batch)
                    prob_cols = [f"Prob_{cls}" for cls in model.classes_]
                    proba_df = pd.DataFrame(probas, columns=prob_cols)
                    df = pd.concat([df, proba_df], axis=1)

                st.success("✅ Predictions complete!")
                st.dataframe(df, use_container_width=True)

                # Option to download results
                csv_download = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Predictions as CSV",
                    csv_download,
                    "predictions.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")
```
