import streamlit as st
import pandas as pd
import numpy as np
import os 
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .hero {
            padding: 30px;
            border-radius: 18px;
            background: linear-gradient(135deg, #1f4e79, #2878b5);
            color: white;
            margin-bottom: 25px;
        }

        .hero h1 {
            font-size: 42px;
            margin-bottom: 8px;
        }

        .hero p {
            font-size: 18px;
            margin: 0;
        }

        .section-title {
            font-size: 24px;
            font-weight: 700;
            margin-top: 15px;
            margin-bottom: 15px;
        }

        .result-box {
            padding: 25px;
            border-radius: 15px;
            background-color: #ffffff;
            border: 1px solid #d9e2ec;
            text-align: center;
            margin-top: 20px;
        }

        .result-price {
            font-size: 38px;
            font-weight: bold;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            font-size: 18px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏠 House Price Predictor</h1>
        <p>Predict the price of a house using machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Enter the property details below and click **Predict House Price**."
)

# ---------------------------------------------------------
# FEATURES USED BY YOUR TRAINED MODEL
# ---------------------------------------------------------
fields = [
    "Order",
    "PID",
    "MS SubClass",
    "Lot Frontage",
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Low Qual Fin SF",
    "Gr Liv Area",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Yr Blt",
    "Garage Cars",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch",
    "Pool Area",
    "Misc Val",
    "Mo Sold",
    "Yr Sold",
]

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">🏡 Property Information</div>',
    unsafe_allow_html=True,
)

data = {}

# Put inputs into columns so the UI isn't extremely long.
columns = st.columns(3)

for i, field in enumerate(fields):
    with columns[i % 3]:
        data[field] = st.number_input(
            field,
            value=0.0,
            format="%.2f",
            key=field,
        )

st.divider()
# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

MODEL_PATH = "models/house_price_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()
# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("🔮 Predict House Price", type="primary"):

    input_df = pd.DataFrame([data])

    if model is None:
        st.error(
            "Model not found. Please run the training pipeline first."
        )
        st.stop()

    try:
        # ---------------------------------------------
        # Apply SAME feature engineering as training
        # ---------------------------------------------

        input_df["Gr Liv Area"] = np.log1p(
            input_df["Gr Liv Area"]
        )

        # ---------------------------------------------
        # Predict log(SalePrice)
        # ---------------------------------------------

        log_prediction = model.predict(input_df)[0]

        # ---------------------------------------------
        # Convert log price back to actual price
        # ---------------------------------------------

        predicted_price = np.expm1(log_prediction)

        # Safety check
        if not np.isfinite(predicted_price) or predicted_price <= 0:
            st.error(
                "The model produced an invalid price. "
                "Please check the property values."
            )
            st.stop()

        # ---------------------------------------------
        # Display result
        # ---------------------------------------------

        st.markdown(
            f"""
            <div class="result-box">
                <div style="font-size:22px;">
                    🏠 Predicted House Price
                </div>
                <div style="font-size:38px; font-weight:bold;">
                    ${predicted_price:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------
        # Save prediction CSV
        # ---------------------------------------------

        csv_df = input_df.copy()

        # Put original Gr Liv Area back in CSV
        csv_df["Gr Liv Area"] = data["Gr Liv Area"]

        csv_df["Predicted Price"] = predicted_price

        csv = csv_df.to_csv(index=False)

        st.download_button(
            "📥 Download Prediction CSV",
            csv,
            "house_price_prediction.csv",
            "text/csv",
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")