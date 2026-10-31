
import streamlit as st
import pandas as pd
import joblib

# Configure the webpage
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# Load the saved model bundle
@st.cache_resource
def load_bundle():
    return joblib.load("iris_model.joblib")

bundle = load_bundle()

model = bundle["model"]
features = bundle["features"]
target_names = bundle["target_names"]
stats = bundle["stats"]

st.title("🌸 Iris Flower Prediction")

st.write(
    "Enter the measurements of an Iris flower "
    "to predict its species using a trained "
    "Support Vector Machine model."
)

st.subheader("Input Features")

# Create input sliders
values = {}

columns = st.columns(2)

for i, feature in enumerate(features):
    info = stats[feature]

    with columns[i % 2]:
        values[feature] = st.slider(
            feature.title(),
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["mean"]),
            step=0.1
        )

# Convert user inputs into a DataFrame
input_df = pd.DataFrame(
    [values],
    columns=features
)

st.subheader("Your Input")
st.dataframe(input_df, hide_index=True)

# Generate prediction when button is clicked
if st.button("Predict", type="primary"):

    # Predict the flower species
    prediction = model.predict(input_df)[0]

    # Calculate probabilities for all three classes
    probabilities = model.predict_proba(input_df)[0]

    # Get the predicted species name
    predicted_name = target_names[prediction]

    # Get probability of the predicted species
    confidence = probabilities[prediction]

    st.success(
        f"Predicted Flower Species: "
        f"{predicted_name.title()}"
    )

    st.metric(
        "Prediction Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.subheader("Class Probabilities")

    for name, probability in zip(
        target_names, probabilities
    ):
        st.write(
            f"{name.title()}: {probability * 100:.2f}%"
        )
        st.progress(float(probability))
