"""
Streamlit App for Wellness Tourism Package Prediction
=====================================================
Loads the trained model from Hugging Face Model Hub and provides
a web interface for predicting customer purchase likelihood.
"""

import streamlit as st
import pandas as pd
import joblib
import json
import os
from huggingface_hub import hf_hub_download

# =========================================================================
# Configuration
# =========================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Auto-detect username from token
if HF_TOKEN:
    from huggingface_hub import HfApi
    try:
        hf_api = HfApi()
        user_info = hf_api.whoami(token=HF_TOKEN)
        HF_USERNAME = user_info["name"]
    except Exception:
        HF_USERNAME = os.environ.get("HF_USERNAME", "WILDESOUL")
else:
    HF_USERNAME = os.environ.get("HF_USERNAME", "WILDESOUL")

MODEL_REPO = f"{HF_USERNAME}/tourism-model"

# =========================================================================
# Load Model
# =========================================================================
@st.cache_resource
def load_model():
    """Load the trained model pipeline and feature info from HF Hub."""
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.joblib",
        token=HF_TOKEN
    )
    features_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="feature_info.json",
        token=HF_TOKEN
    )

    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        feature_info = json.load(f)

    return model, feature_info

# =========================================================================
# Page Configuration
# =========================================================================
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌴",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        margin-top: 20px;
    }
    .stButton>button {
        width: 100%;
        height: 50px;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# App Header
# =========================================================================
st.markdown('<p class="main-header">🌴 Wellness Tourism Package Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether a customer will purchase the Wellness Tourism Package</p>', unsafe_allow_html=True)
st.markdown("---")

# Load model
try:
    model, feature_info = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model_loaded = False

if model_loaded:
    # =========================================================================
    # Input Form
    # =========================================================================
    st.subheader("📋 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Details**")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Age of the customer")
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=100000, value=20000)

    with col2:
        st.markdown("**Travel Preferences**")
        city_tier = st.selectbox("City Tier", [1, 2, 3], help="City category (Tier 1 > Tier 2 > Tier 3)")
        num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)
        num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
        preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        num_trips = st.number_input("Average Annual Trips", min_value=1.0, max_value=25.0, value=2.0, step=1.0)
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col3:
        st.markdown("**Interaction Details**")
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=1.0, max_value=60.0, value=15.0)
        num_followups = st.number_input("Number of Follow-ups", min_value=1.0, max_value=6.0, value=3.0, step=1.0)
        pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

    st.markdown("---")

    # =========================================================================
    # Prediction
    # =========================================================================
    if st.button("🔮 Predict Purchase Likelihood", type="primary"):
        # Create input dataframe matching training feature order
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [type_of_contact],
            'CityTier': [city_tier],
            'DurationOfPitch': [duration_of_pitch],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [num_person_visiting],
            'NumberOfFollowups': [num_followups],
            'ProductPitched': [product_pitched],
            'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_status],
            'NumberOfTrips': [num_trips],
            'Passport': [passport],
            'PitchSatisfactionScore': [pitch_satisfaction_score],
            'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [num_children_visiting],
            'Designation': [designation],
            'MonthlyIncome': [monthly_income]
        })

        # Reorder columns to match training order
        feature_names = feature_info['feature_names']
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        col_result1, col_result2 = st.columns(2)

        with col_result1:
            if prediction == 1:
                st.success(f"✅ **LIKELY TO PURCHASE**")
                st.metric("Purchase Probability", f"{probability[1]:.1%}")
            else:
                st.error(f"❌ **UNLIKELY TO PURCHASE**")
                st.metric("Purchase Probability", f"{probability[1]:.1%}")

        with col_result2:
            st.markdown("**Probability Distribution**")
            prob_df = pd.DataFrame({
                'Outcome': ['Will Not Purchase', 'Will Purchase'],
                'Probability': [probability[0], probability[1]]
            })
            st.bar_chart(prob_df.set_index('Outcome'))

    # =========================================================================
    # Model Info
    # =========================================================================
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Model Type:** {feature_info.get('best_model_name', 'N/A')}")
        st.write(f"**Best F1 Score:** {feature_info.get('best_f1_score', 'N/A'):.4f}")
        st.write(f"**Model Repository:** [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})")
        st.write(f"**Features Used:** {len(feature_info.get('feature_names', []))}")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #999;">Built by Visit with Us | Powered by Streamlit & Hugging Face 🤗</p>',
    unsafe_allow_html=True
)
