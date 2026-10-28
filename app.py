from pathlib import Path
import joblib
import streamlit as st

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="centered")

MODEL_PATH = Path(__file__).parent / "models" / "sentiment_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.title("💬 AI-Powered Sentiment Analysis")
st.write("Enter a customer review and classify it as **Positive**, **Negative**, or **Neutral**.")
review = st.text_area("Customer review", placeholder="Example: The product works well and the quality is excellent.", height=140)

if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        prediction = model.predict([review.strip()])[0]
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([review.strip()])[0]
            classes = model.classes_
            confidence = float(max(probabilities))
            st.subheader(f"Predicted sentiment: {prediction}")
            st.metric("Model probability for predicted class", f"{confidence:.1%}")
            st.caption("This is the model's estimated class probability, not a guarantee that the prediction is correct.")
            st.bar_chart({str(c): float(p) for c, p in zip(classes, probabilities)})
        else:
            st.subheader(f"Predicted sentiment: {prediction}")
            st.info("This model does not provide calibrated class probabilities.")

st.divider()
st.caption("Built with TF-IDF and a scikit-learn classification pipeline.")
