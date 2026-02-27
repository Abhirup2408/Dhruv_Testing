import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Paddy Price Predictor",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Paddy Price Prediction App")
st.markdown("Enter paddy quality parameters to predict the market price.")

# ==========================================================
# LOAD TRAINED XGBOOST MODEL
# ==========================================================

MODEL_PATH = "price_model_advanced.pkl"

@st.cache_resource
def load_model():
    with open("price_model_advanced.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data["model"], model_data["features"]

model, features = load_model()

# ==========================================================
# USER INPUT SECTION
# ==========================================================

st.subheader("📌 Enter Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    after_length = st.number_input(
        "After Length",
        min_value=0.0,
        placeholder="Enter if available"
    )

    after_mic = st.number_input(
        "After Moisture (after_mic)",
        min_value=0.0,
        placeholder="Enter if available"
    )

with col2:
    lab_length = st.number_input(
        "Lab Length",
        min_value=0.0,
        placeholder="Enter if available"
    )

    lab_mic = st.number_input(
        "Lab Moisture (lab_mic)",
        min_value=0.0,
        placeholder="Enter if available"
    )

mpp = st.number_input(
    "MPP",
    min_value=0.0,
    placeholder="Enter MPP value"
)

recovery = st.number_input(
    "Recovery %",
    min_value=0.0,
    placeholder="Enter Recovery %"
)

# ==========================================================
# PREDICTION BUTTON
# ==========================================================

if st.button("🔍 Predict Price"):

    # -------------------------
    # VALIDATION
    # -------------------------

    if after_length == 0 and lab_length == 0:
        st.error("⚠ Please enter at least one Length value.")
        st.stop()

    if after_mic == 0 and lab_mic == 0:
        st.error("⚠ Please enter at least one Moisture value.")
        st.stop()

    if mpp == 0:
        st.error("⚠ Please enter MPP value.")
        st.stop()

    if recovery == 0:
        st.error("⚠ Please enter Recovery value.")
        st.stop()

    # -------------------------
    # DYNAMIC SELECTION
    # -------------------------

    final_length = after_length if after_length > 0 else lab_length
    final_moisture = after_mic if after_mic > 0 else lab_mic
    final_recovery = recovery

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------

    length_x_mpp = final_length * mpp
    moisture_x_recovery = final_moisture * final_recovery
    length_div_moisture = final_length / (final_moisture + 1e-6)

    input_data = pd.DataFrame([[
        final_length,
        final_moisture,
        mpp,
        final_recovery,
        length_x_mpp,
        moisture_x_recovery,
        length_div_moisture
    ]], columns=features)

    # -------------------------
    # PREDICTION
    # -------------------------

    prediction_log = model.predict(input_data)[0]

    # Reverse log transform
    predicted_price = np.expm1(prediction_log)

    # -------------------------
    # DISPLAY RESULT
    # -------------------------

    st.success(f"💰 Predicted Price: ₹ {round(predicted_price, 2)}")

    # st.subheader("📊 Prediction Details")
    # st.write(input_data)
