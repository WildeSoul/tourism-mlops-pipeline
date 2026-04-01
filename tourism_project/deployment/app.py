"""
Streamlit App — Wellness Tourism Package Predictor (Advanced Dashboard)
=========================================================================
Multi-page dashboard with:
  1. Single Prediction — with SHAP explanation + risk scoring
  2. Batch Prediction — CSV upload with downloadable results
  3. Model Analytics — comparison table, SHAP plots, confusion matrix, ROC
  4. About — architecture, pipeline explanation, model card
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
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
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"


# =========================================================================
# Load Model & Artifacts
# =========================================================================
@st.cache_resource
def load_model():
    """Load the trained model pipeline and feature info from HF Hub."""
    model_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="best_model.joblib", token=HF_TOKEN
    )
    features_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="feature_info.json", token=HF_TOKEN
    )
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        feature_info = json.load(f)
    return model, feature_info


@st.cache_data
def load_model_comparison():
    """Load model comparison results."""
    try:
        path = hf_hub_download(
            repo_id=MODEL_REPO, filename="model_comparison.json", token=HF_TOKEN
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data
def load_plot(filename):
    """Load a plot image from HF Hub."""
    try:
        path = hf_hub_download(
            repo_id=MODEL_REPO, filename=f"plots/{filename}", token=HF_TOKEN
        )
        return path
    except Exception:
        return None


def get_feature_names():
    """Get the complete list of feature names for the model."""
    return [
        'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
        'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched',
        'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport',
        'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
        'Designation', 'MonthlyIncome',
        'IncomePerPerson', 'TotalVisitors', 'HighDesignation',
        'IsFrequentTraveler', 'HasChildren'
    ]


def engineer_input_features(df):
    """Apply feature engineering to input data (match training pipeline)."""
    df = df.copy()
    df['IncomePerPerson'] = df['MonthlyIncome'] / df['NumberOfPersonVisiting'].clip(lower=1)
    df['TotalVisitors'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
    df['HighDesignation'] = df['Designation'].isin(['AVP', 'VP']).astype(int)
    df['IsFrequentTraveler'] = (df['NumberOfTrips'] > 3).astype(int)
    df['HasChildren'] = (df['NumberOfChildrenVisiting'] > 0).astype(int)
    return df


def get_risk_level(probability):
    """Classify prediction probability into risk levels."""
    if probability >= 0.75:
        return "🟢 HIGH", "High Likelihood", "#2ecc71"
    elif probability >= 0.50:
        return "🟡 MEDIUM", "Moderate Likelihood", "#f39c12"
    elif probability >= 0.25:
        return "🟠 LOW", "Low Likelihood", "#e67e22"
    else:
        return "🔴 VERY LOW", "Very Low Likelihood", "#e74c3c"


# =========================================================================
# Page Configuration
# =========================================================================
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #1E88E5, #00ACC1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        font-weight: 800;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 5px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .risk-badge {
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        height: 50px;
        font-size: 1.1rem;
        border-radius: 25px;
        font-weight: 600;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Sidebar Navigation
# =========================================================================
with st.sidebar:
    st.markdown("## 🌴 Navigation")
    page = st.radio(
        "Select Page",
        ["🔮 Single Prediction", "📦 Batch Prediction", "📊 Model Analytics", "ℹ️ About"],
        index=0
    )
    st.markdown("---")
    st.markdown("### 🔧 Model Info")
    try:
        _, fi = load_model()
        st.success(f"**Model:** {fi.get('best_model_name', 'N/A')}")
        st.info(f"**F1 Score:** {fi.get('best_f1_score', 0):.4f}")
        st.info(f"**Features:** {len(fi.get('feature_names', []))}")
        if fi.get('smote_applied'):
            st.info("**SMOTE:** Applied ✅")
    except Exception:
        st.warning("Model not loaded yet")

    st.markdown("---")
    st.markdown(
        f"[🤗 Model Hub](https://huggingface.co/{MODEL_REPO})",
        unsafe_allow_html=True
    )


# =========================================================================
# Page 1: Single Prediction
# =========================================================================
if page == "🔮 Single Prediction":
    st.markdown('<p class="main-header">🌴 Wellness Tourism Package Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether a customer will purchase the Wellness Tourism Package</p>', unsafe_allow_html=True)
    st.markdown("---")

    try:
        model, feature_info = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        model_loaded = False

    if model_loaded:
        st.subheader("📋 Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Personal Details**")
            age = st.number_input("Age", min_value=18, value=30, help="Customer age")
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, value=20000)

        with col2:
            st.markdown("**✈️ Travel Preferences**")
            city_tier = st.selectbox("City Tier", [1, 2, 3], help="City category")
            num_person_visiting = st.number_input("Persons Visiting", min_value=1, value=2)
            num_children_visiting = st.number_input("Children Visiting", min_value=0, value=0)
            preferred_property_star = st.selectbox("Property Star", [3.0, 4.0, 5.0])
            num_trips = st.number_input("Annual Trips", min_value=1.0, value=2.0, step=1.0)
            passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
            own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")

        with col3:
            st.markdown("**📞 Interaction Details**")
            type_of_contact = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            duration_of_pitch = st.number_input("Pitch Duration (min)", min_value=1.0, value=15.0)
            num_followups = st.number_input("Follow-ups", min_value=1.0, value=3.0, step=1.0)
            pitch_satisfaction_score = st.selectbox("Pitch Satisfaction", [1, 2, 3, 4, 5])

        st.markdown("---")

        if st.button("🔮 Predict Purchase Likelihood", type="primary"):
            # Build input dataframe
            input_data = pd.DataFrame({
                'Age': [age], 'TypeofContact': [type_of_contact], 'CityTier': [city_tier],
                'DurationOfPitch': [duration_of_pitch], 'Occupation': [occupation],
                'Gender': [gender], 'NumberOfPersonVisiting': [num_person_visiting],
                'NumberOfFollowups': [num_followups], 'ProductPitched': [product_pitched],
                'PreferredPropertyStar': [preferred_property_star],
                'MaritalStatus': [marital_status], 'NumberOfTrips': [num_trips],
                'Passport': [passport], 'PitchSatisfactionScore': [pitch_satisfaction_score],
                'OwnCar': [own_car], 'NumberOfChildrenVisiting': [num_children_visiting],
                'Designation': [designation], 'MonthlyIncome': [monthly_income]
            })

            # Apply feature engineering
            input_data = engineer_input_features(input_data)

            # Reorder columns
            feature_names = feature_info['feature_names']
            input_data = input_data.reindex(columns=feature_names, fill_value=0)

            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            purchase_prob = probability[1]

            # Risk level
            risk_icon, risk_label, risk_color = get_risk_level(purchase_prob)

            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            col_r1, col_r2, col_r3 = st.columns([1, 1, 1])

            with col_r1:
                if prediction == 1:
                    st.success("### ✅ LIKELY TO PURCHASE")
                else:
                    st.error("### ❌ UNLIKELY TO PURCHASE")

            with col_r2:
                st.metric("Purchase Probability", f"{purchase_prob:.1%}")

            with col_r3:
                st.markdown(f"""
                <div style="background-color: {risk_color}22; border: 2px solid {risk_color};
                            border-radius: 12px; padding: 15px; text-align: center;">
                    <span style="font-size: 1.5rem;">{risk_icon}</span><br>
                    <span style="font-size: 1rem; color: {risk_color}; font-weight: bold;">{risk_label}</span>
                </div>
                """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("#### 📊 Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)

            with prob_col1:
                import plotly.graph_objects as go
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=purchase_prob * 100,
                    title={'text': "Purchase Likelihood", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': risk_color},
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0, 25], 'color': '#fadbd8'},
                            {'range': [25, 50], 'color': '#fdebd0'},
                            {'range': [50, 75], 'color': '#fef9e7'},
                            {'range': [75, 100], 'color': '#d5f5e3'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=40, b=0, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)

            with prob_col2:
                prob_df = pd.DataFrame({
                    'Outcome': ['Will Not Purchase', 'Will Purchase'],
                    'Probability': [probability[0], probability[1]]
                })
                st.bar_chart(prob_df.set_index('Outcome'))

            # Download prediction report
            report = {
                'prediction': 'Purchase' if prediction == 1 else 'No Purchase',
                'probability': f"{purchase_prob:.4f}",
                'risk_level': risk_label,
                'customer_age': age,
                'customer_income': monthly_income,
                'model_used': feature_info.get('best_model_name', 'N/A')
            }
            report_csv = pd.DataFrame([report])
            st.download_button(
                "📥 Download Prediction Report",
                data=report_csv.to_csv(index=False),
                file_name="prediction_report.csv",
                mime="text/csv"
            )


# =========================================================================
# Page 2: Batch Prediction
# =========================================================================
elif page == "📦 Batch Prediction":
    st.markdown('<p class="main-header">📦 Batch Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a CSV file to predict purchase likelihood for multiple customers</p>', unsafe_allow_html=True)
    st.markdown("---")

    try:
        model, feature_info = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        model_loaded = False

    if model_loaded:
        # Sample CSV template
        st.markdown("### 📄 Upload Customer Data")
        st.info("""
        **Required Columns:** Age, TypeofContact, CityTier, DurationOfPitch, Occupation,
        Gender, NumberOfPersonVisiting, NumberOfFollowups, ProductPitched,
        PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport,
        PitchSatisfactionScore, OwnCar, NumberOfChildrenVisiting, Designation, MonthlyIncome
        """)

        # Download template
        template_df = pd.DataFrame({
            'Age': [30, 45],
            'TypeofContact': ['Self Enquiry', 'Company Invited'],
            'CityTier': [1, 2],
            'DurationOfPitch': [15.0, 20.0],
            'Occupation': ['Salaried', 'Small Business'],
            'Gender': ['Male', 'Female'],
            'NumberOfPersonVisiting': [2, 3],
            'NumberOfFollowups': [3.0, 4.0],
            'ProductPitched': ['Deluxe', 'King'],
            'PreferredPropertyStar': [4.0, 5.0],
            'MaritalStatus': ['Married', 'Single'],
            'NumberOfTrips': [3.0, 5.0],
            'Passport': [1, 0],
            'PitchSatisfactionScore': [4, 3],
            'OwnCar': [1, 0],
            'NumberOfChildrenVisiting': [1, 0],
            'Designation': ['Manager', 'VP'],
            'MonthlyIncome': [25000, 45000]
        })

        st.download_button(
            "📥 Download CSV Template",
            data=template_df.to_csv(index=False),
            file_name="batch_prediction_template.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df)} records")
                st.dataframe(batch_df.head(), use_container_width=True)

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Apply feature engineering
                        batch_engineered = engineer_input_features(batch_df)

                        # Reorder columns
                        feature_names = feature_info['feature_names']
                        batch_engineered = batch_engineered.reindex(columns=feature_names, fill_value=0)

                        # Predict
                        predictions = model.predict(batch_engineered)
                        probabilities = model.predict_proba(batch_engineered)[:, 1]

                        # Add results
                        results_df = batch_df.copy()
                        results_df['Prediction'] = ['Purchase' if p == 1 else 'No Purchase' for p in predictions]
                        results_df['Purchase_Probability'] = probabilities.round(4)
                        results_df['Risk_Level'] = [get_risk_level(p)[1] for p in probabilities]

                    st.markdown("### 📊 Batch Results")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    total = len(results_df)
                    likely = (predictions == 1).sum()
                    unlikely = (predictions == 0).sum()
                    avg_prob = probabilities.mean()

                    col1.metric("Total Customers", total)
                    col2.metric("Likely to Purchase", f"{likely} ({likely/total*100:.1f}%)")
                    col3.metric("Unlikely to Purchase", f"{unlikely} ({unlikely/total*100:.1f}%)")
                    col4.metric("Avg Probability", f"{avg_prob:.1%}")

                    st.markdown("---")
                    st.dataframe(
                        results_df.style.applymap(
                            lambda x: 'background-color: #d5f5e3' if x == 'Purchase' else
                                      'background-color: #fadbd8' if x == 'No Purchase' else '',
                            subset=['Prediction']
                        ),
                        use_container_width=True
                    )

                    # Distribution chart
                    import plotly.express as px
                    fig = px.histogram(
                        results_df, x='Purchase_Probability', nbins=20,
                        title='Distribution of Purchase Probabilities',
                        color_discrete_sequence=['#2980b9']
                    )
                    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                                 annotation_text="Decision Threshold")
                    st.plotly_chart(fig, use_container_width=True)

                    # Download results
                    st.download_button(
                        "📥 Download Batch Results",
                        data=results_df.to_csv(index=False),
                        file_name="batch_prediction_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")


# =========================================================================
# Page 3: Model Analytics
# =========================================================================
elif page == "📊 Model Analytics":
    st.markdown('<p class="main-header">📊 Model Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive model performance analysis and explainability</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Model Comparison Table
    comparison = load_model_comparison()
    if comparison:
        st.subheader("🏆 Model Comparison")
        model_results = comparison.get('model_results', {})
        if model_results:
            comp_df = pd.DataFrame(model_results).T
            comp_df = comp_df.round(4)
            comp_df = comp_df.sort_values('f1_score', ascending=False)
            best = comparison.get('best_model', '')

            # Highlight best model
            st.dataframe(
                comp_df.style.highlight_max(axis=0, color='#d5f5e3'),
                use_container_width=True
            )

            st.success(f"🏆 **Best Model:** {best} (F1: {comparison.get('best_f1_score', 0):.4f})")

            # Training config
            config = comparison.get('training_config', {})
            if config:
                cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns(4)
                cfg_col1.metric("SMOTE Applied", "✅ Yes" if config.get('smote') else "❌ No")
                cfg_col2.metric("CV Strategy", config.get('cv_strategy', 'N/A'))
                cfg_col3.metric("Scoring Metric", config.get('scoring', 'N/A'))
                cfg_col4.metric("Models Trained", config.get('n_models', 0))
    else:
        st.warning("Model comparison data not available")

    st.markdown("---")

    # Visualizations
    st.subheader("📈 Performance Visualizations")

    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "Model Comparison", "Confusion Matrix", "ROC Curves", "SHAP Summary", "SHAP Importance"
    ])

    with viz_tab1:
        path = load_plot("model_comparison_chart.png")
        if path:
            st.image(path, caption="Model Comparison — F1 Score, AUC-ROC, Accuracy", use_column_width=True)
        else:
            st.info("Model comparison chart not available")

    with viz_tab2:
        path = load_plot("confusion_matrix.png")
        if path:
            st.image(path, caption="Confusion Matrix — Best Model", use_column_width=True)
        else:
            st.info("Confusion matrix plot not available")

    with viz_tab3:
        path = load_plot("roc_curves_comparison.png")
        if path:
            st.image(path, caption="ROC Curves — All Models Comparison", use_column_width=True)
        else:
            path2 = load_plot("roc_curve.png")
            if path2:
                st.image(path2, caption="ROC Curve — Best Model", use_column_width=True)
            else:
                st.info("ROC curve plot not available")

    with viz_tab4:
        path = load_plot("shap_summary_plot.png")
        if path:
            st.image(path, caption="SHAP Feature Impact Summary (Beeswarm)", use_column_width=True)
            st.markdown("""
            **How to read this plot:**
            - Each dot represents a single prediction
            - Red = high feature value, Blue = low feature value
            - Dots to the right push the prediction toward **Purchase**
            - Dots to the left push the prediction toward **No Purchase**
            """)
        else:
            st.info("SHAP summary plot not available")

    with viz_tab5:
        path = load_plot("shap_feature_importance.png")
        if path:
            st.image(path, caption="SHAP Feature Importance (Mean |SHAP| Value)", use_column_width=True)
        else:
            st.info("SHAP feature importance plot not available")


# =========================================================================
# Page 4: About
# =========================================================================
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ## 🏢 Business Context

    **"Visit with Us"**, a leading travel company, is revolutionizing the tourism industry
    by leveraging data-driven strategies to optimize operations and customer engagement.
    This system predicts whether customers will purchase the **Wellness Tourism Package**
    before contacting them.

    ---

    ## 🏗️ Architecture

    ```
    ┌──────────────────────────────────────────────────────────┐
    │                    GitHub Actions CI/CD                    │
    │                                                           │
    │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
    │  │ Register │→ │   Data   │→ │  Model   │→ │  Deploy   │ │
    │  │ Dataset  │  │   Prep   │  │ Training │  │  to HF    │ │
    │  └─────────┘  └──────────┘  └──────────┘  └───────────┘ │
    │       │            │             │              │         │
    │       │       ┌────────┐   ┌──────────┐  ┌───────────┐  │
    │       │       │Validate│   │ Quality  │  │ Streamlit │  │
    │       │       │  Data  │   │   Gate   │  │    App    │  │
    │       │       └────────┘   └──────────┘  └───────────┘  │
    └──────────────────────────────────────────────────────────┘
    ```

    ---

    ## 🔬 Advanced Features

    | Feature | Description |
    |---------|-------------|
    | **EDA Visualizations** | 6 publication-quality charts (correlation, distributions, boxplots) |
    | **Feature Engineering** | 5 derived features (IncomePerPerson, TotalVisitors, etc.) |
    | **Outlier Detection** | IQR-based capping on key numerical features |
    | **SMOTE** | Synthetic Minority Oversampling for class imbalance |
    | **6 ML Models** | DT, RF, GB, XGBoost, AdaBoost, LightGBM |
    | **SHAP Explainability** | Global feature importance + individual prediction explanations |
    | **Model Quality Gate** | F1 ≥ 0.70 and AUC ≥ 0.75 thresholds before deployment |
    | **Data Validation** | Schema, null, and class balance checks in CI/CD |
    | **Batch Prediction** | CSV upload for bulk customer scoring |
    | **MLflow Tracking** | Full experiment tracking with artifacts |

    ---

    ## 📊 Models Trained

    1. **Decision Tree** — Baseline interpretable model
    2. **Random Forest** — Ensemble of decision trees
    3. **Gradient Boosting** — Sequential boosted trees
    4. **XGBoost** — Optimized gradient boosting
    5. **AdaBoost** — Adaptive boosting
    6. **LightGBM** — Microsoft's fast gradient boosting

    All models are tuned with **RandomizedSearchCV** and **StratifiedKFold(5)** cross-validation.

    ---

    ## 🔗 Links

    - **Model Repository:** [Hugging Face](https://huggingface.co/{MODEL_REPO})
    - **Dataset Repository:** [Hugging Face](https://huggingface.co/datasets/{DATASET_REPO})
    """.replace("{MODEL_REPO}", MODEL_REPO).replace("{DATASET_REPO}", DATASET_REPO))


# =========================================================================
# Footer
# =========================================================================
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #999; font-size: 0.9rem;">'
    'Built by Visit with Us | Powered by Streamlit & Hugging Face 🤗 | '
    'Advanced MLOps Pipeline with SHAP, SMOTE & 6 ML Models'
    '</p>',
    unsafe_allow_html=True
)
