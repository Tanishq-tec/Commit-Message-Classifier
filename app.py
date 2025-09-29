import streamlit as st
import pandas as pd
import joblib
import sklearn  # Ensure sklearn is available

# ----------- Load Model and Vectorizer with caching -----------
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
st.write("Predict the quality of commit messages (Low vs High).")

# ----------- Tabs for Single / Batch Prediction -----------
tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch Prediction (CSV)"])

# ----------- Single Prediction -----------
with tab1:
    st.subheader("🔹 Classify a Single Commit Message")
    user_input = st.text_area("Commit Message:", placeholder="Type a commit message here...")

    if st.button("Classify", key="single"):
        if user_input.strip():
            try:
                # Preprocess
                cleaned = user_input.lower()
                X = vectorizer.transform([cleaned])
                prediction = model.predict(X)[0]

                # Map numeric label to readable
                label_map = {0: "Low Quality", 1: "High Quality"}
                st.success(f"**Prediction:** {label_map[prediction]}")

                # Show probabilities if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    prob_df = pd.DataFrame({
                        "Class": [label_map[i] for i in model.classes_],
                        "Probability": proba
                    }).sort_values("Probability", ascending=False)
                    st.write("### Class Probabilities")
                    st.dataframe(prob_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
        else:
            st.warning("⚠️ Please enter a commit message before classifying.")

# ----------- Batch Prediction (CSV Upload) -----------
with tab2:
    st.subheader("📂 Upload a CSV file for Batch Prediction")
    st.write("CSV must contain a column named `commit_message`.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "commit_message" not in df.columns:
                st.error("❌ CSV must contain a column named `commit_message`.")
            else:
                # Preprocess
                df["cleaned"] = df["commit_message"].str.lower()
                X_batch = vectorizer.transform(df["cleaned"])
                preds = model.predict(X_batch)
                label_map = {0: "Low Quality", 1: "High Quality"}
                df["Prediction"] = [label_map[p] for p in preds]

                # Add probabilities if available
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_batch)
                    prob_cols = [f"Prob_{label_map[c]}" for c in model.classes_]
                    proba_df = pd.DataFrame(probas, columns=prob_cols)
                    df = pd.concat([df, prob_df], axis=1)

                st.success("✅ Predictions complete!")
                st.dataframe(df, use_container_width=True)

                # Download results
                csv_download = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Predictions as CSV",
                    csv_download,
                    "predictions.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")
