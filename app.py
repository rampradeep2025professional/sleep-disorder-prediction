"""Sleep Disorder Classification Web Application.

A modern, responsive, and educational machine learning application built with Streamlit
to classify sleep disorder categories based on lifestyle, sleep, and cardiovascular metrics.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Sleep Disorder Classification",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Lightweight Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        color: #0F172A;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .pred-none {
        background-color: #10B981;
    }
    .pred-insomnia {
        background-color: #F59E0B;
    }
    .pred-apnea {
        background-color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)

# Required Disclaimer Text
MANDATORY_DISCLAIMER = (
    "“This application is developed for educational and machine-learning demonstration "
    "purposes only. Its predictions are based on patterns in the training dataset and are not "
    "a medical diagnosis. Model performance may be limited by dataset quality, class imbalance, "
    "and differences between training data and real-world users. Consult a qualified healthcare "
    "professional for medical concerns.”"
)

MODEL_FILE = os.path.join("model", "sleep_disorder_pipeline.joblib")
METRICS_FILE = os.path.join("model", "model_metrics.json")
COMPARISON_FILE = os.path.join("model", "model_comparison.csv")
DATA_FILE = os.path.join("data", "sleep_health.csv")


@st.cache_resource
def load_model():
    """Loads the trained scikit-learn pipeline."""
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            st.error(f"Error loading model artifact: {e}")
            return None
    return None


@st.cache_data
def load_metrics():
    """Loads evaluated model metrics and comparison results."""
    metrics = None
    comparison_df = None
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    if os.path.exists(COMPARISON_FILE):
        try:
            comparison_df = pd.read_csv(COMPARISON_FILE)
        except Exception:
            comparison_df = None
    return metrics, comparison_df


@st.cache_data
def load_dataset_sample():
    """Loads dataset for visualization and statistics."""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None").astype(str).str.strip()
            df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
            return df
        except Exception:
            return None
    return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/sleeping-in-bed.png", width=70)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Predict", "Dataset Insights", "Model Performance", "About Project"],
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Mini-Project:** Sleep Disorder Classification Using Machine Learning")
st.sidebar.caption("Supervised multi-class classification using Scikit-Learn and Streamlit.")


# ==========================================
# PAGE 1: HOME
# ==========================================
if page == "Home":
    st.markdown("<div class='main-header'>Sleep Disorder Classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Machine Learning-Based Sleep Health Classification</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Welcome to the Sleep Health ML Classifier
        Sleep is a critical foundation for cardiovascular, neurological, and emotional well-being.
        This application leverages machine learning algorithms trained on clinical and lifestyle attributes
        to classify sleep status into one of three primary outcomes:
        
        - 🟢 **No Sleep Disorder (None)**: Healthy sleep duration and architecture without chronic pathology.
        - 🟠 **Insomnia**: Difficulty falling or staying asleep, often associated with elevated stress and reduced duration.
        - 🔴 **Sleep Apnea**: Repeated pauses in breathing during sleep, strongly linked to elevated blood pressure and BMI.
        
        #### Key Project Highlights:
        - **End-to-End Pipeline**: Unified feature engineering, scaling, and classification to eliminate data leakage.
        - **Robust Preprocessing**: Automatic standardization of BMI categories and physiological decomposition of blood pressure into systolic/diastolic components.
        - **Model Comparison**: Benchmarked Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Gradient Boosting.
        - **Educational Transparency**: Full access to exploratory charts, confusion matrices, and cross-validated metrics.
        """)
        st.info("👉 Use the **Predict** tab in the sidebar to test predictions with custom lifestyle inputs, or explore **Dataset Insights** to view data distributions.")
    
    with col2:
        st.markdown("#### System At A Glance")
        metrics_data, _ = load_metrics()
        if metrics_data:
            best_model_name = metrics_data.get("best_model", "Random Forest")
            test_acc = metrics_data["summary_metrics"].get("Test Accuracy", 0.0) * 100
            test_f1 = metrics_data["summary_metrics"].get("Test Macro F1", 0.0) * 100
            
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom: 12px;'>
                <div class='metric-title'>Selected Production Model</div>
                <div class='metric-value'>{best_model_name}</div>
            </div>
            <div class='metric-card' style='margin-bottom: 12px;'>
                <div class='metric-title'>Test Set Accuracy</div>
                <div class='metric-value'>{test_acc:.1f}%</div>
            </div>
            <div class='metric-card' style='margin-bottom: 12px;'>
                <div class='metric-title'>Macro F1-Score</div>
                <div class='metric-value'>{test_f1:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Model metrics file not detected. Please run `train_model.py`.")

    # Educational Disclaimer Box
    st.markdown(f"<div class='disclaimer-box'><b>Educational Notice:</b><br>{MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 2: PREDICT
# ==========================================
elif page == "Predict":
    st.markdown("<div class='main-header'>Sleep Disorder Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Enter participant lifestyle and physiological parameters to evaluate sleep health status.</div>", unsafe_allow_html=True)

    pipeline = load_model()
    if pipeline is None:
        st.error("⚠️ The trained model pipeline was not found at `model/sleep_disorder_pipeline.joblib`. Please execute `python train_model.py` in your terminal to train and save the pipeline.")
    else:
        st.markdown("#### Patient / Participant Details Form")
        
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("👤 Demographics & Habits")
                gender = st.selectbox("Gender", options=["Male", "Female"], index=0, help="Biological sex of the participant")
                age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35, step=1)
                occupation = st.selectbox(
                    "Occupation",
                    options=[
                        "Doctor", "Nurse", "Engineer", "Lawyer", "Teacher",
                        "Accountant", "Salesperson", "Software Engineer",
                        "Scientist", "Sales Representative", "Manager"
                    ],
                    index=2,
                    help="Primary professional occupation"
                )
                bmi_category = st.selectbox(
                    "BMI Category",
                    options=["Normal", "Overweight", "Obese"],
                    index=0,
                    help="Standardized Body Mass Index classification"
                )
                physical_activity = st.slider(
                    "Physical Activity Level (min/day)",
                    min_value=10, max_value=120, value=60, step=5,
                    help="Average daily duration of moderate-to-vigorous physical exercise"
                )
                daily_steps = st.number_input(
                    "Daily Steps",
                    min_value=1000, max_value=25000, value=7500, step=500,
                    help="Average daily step count recorded via pedometer or wearable"
                )

            with c2:
                st.subheader("💤 Sleep & Cardiovascular")
                sleep_duration = st.slider(
                    "Sleep Duration (Hours / Day)",
                    min_value=3.0, max_value=12.0, value=7.2, step=0.1,
                    help="Average hours of sleep per night"
                )
                quality_of_sleep = st.slider(
                    "Quality of Sleep (Rating 1 - 10)",
                    min_value=1, max_value=10, value=7, step=1,
                    help="Subjective sleep restoration and uninterrupted sleep quality score"
                )
                stress_level = st.slider(
                    "Stress Level (Rating 1 - 10)",
                    min_value=1, max_value=10, value=5, step=1,
                    help="Perceived psychological stress rating"
                )
                heart_rate = st.number_input(
                    "Resting Heart Rate (BPM)",
                    min_value=45, max_value=120, value=70, step=1,
                    help="Resting pulse in beats per minute"
                )
                
                st.markdown("**Blood Pressure (mmHg)**")
                bp_col1, bp_col2 = st.columns(2)
                with bp_col1:
                    systolic = st.number_input("Systolic (Upper)", min_value=80, max_value=200, value=120, step=1)
                with bp_col2:
                    diastolic = st.number_input("Diastolic (Lower)", min_value=50, max_value=130, value=80, step=1)
                bp_string = f"{systolic}/{diastolic}"
                st.caption(f"Synthesized blood pressure: **{bp_string} mmHg**")

            submitted = st.form_submit_button("Predict Sleep Disorder", type="primary", use_container_width=True)

        if submitted:
            # Construct single-row DataFrame matching the training schema exactly
            input_dict = {
                "Gender": [gender],
                "Age": [int(age)],
                "Occupation": [occupation],
                "Sleep Duration": [float(sleep_duration)],
                "Quality of Sleep": [int(quality_of_sleep)],
                "Physical Activity Level": [int(physical_activity)],
                "Stress Level": [int(stress_level)],
                "BMI Category": [bmi_category],
                "Blood Pressure": [bp_string],
                "Heart Rate": [int(heart_rate)],
                "Daily Steps": [int(daily_steps)],
            }
            input_df = pd.DataFrame(input_dict)

            try:
                # Generate model prediction
                prediction = pipeline.predict(input_df)[0]
                has_proba = hasattr(pipeline, "predict_proba")
                probabilities = pipeline.predict_proba(input_df)[0] if has_proba else None
                classes = list(pipeline.classes_) if has_proba else []

                st.markdown("### Prediction Outcome")
                
                # Visual Alert Card
                if prediction == "None":
                    st.markdown("""
                    <div class='prediction-card pred-none'>
                        <h2>🟢 No Sleep Disorder Detected</h2>
                        <p style='font-size: 1.1rem; margin-bottom: 0;'>The model predicts standard, healthy sleep patterns based on your reported metrics.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction == "Insomnia":
                    st.markdown("""
                    <div class='prediction-card pred-insomnia'>
                        <h2>🟠 Prediction: Insomnia</h2>
                        <p style='font-size: 1.1rem; margin-bottom: 0;'>The model identified patterns characteristic of Insomnia (e.g. reduced sleep duration, elevated stress).</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='prediction-card pred-apnea'>
                        <h2>🔴 Prediction: Sleep Apnea</h2>
                        <p style='font-size: 1.1rem; margin-bottom: 0;'>The model identified patterns characteristic of Sleep Apnea (e.g. elevated blood pressure and BMI indicators).</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probabilities chart if supported
                if has_proba and probabilities is not None:
                    st.markdown("#### Model Prediction Confidence Breakdown")
                    proba_df = pd.DataFrame({
                        "Category": classes,
                        "Probability (%)": [p * 100 for p in probabilities],
                    }).set_index("Category")

                    pcol1, pcol2 = st.columns([3, 2])
                    with pcol1:
                        st.bar_chart(proba_df, horizontal=True)
                    with pcol2:
                        st.markdown("**Per-Class Probabilities:**")
                        for cls_name, prob_val in zip(classes, probabilities):
                            st.write(f"- **{cls_name}**: `{prob_val * 100:.2f}%`")
                        st.caption("Note: Probability reflects the decision function distribution across the training dataset.")

                # Clinical Context & Health Information
                with st.expander("ℹ️ Understanding This Prediction & Key Features", expanded=True):
                    st.markdown("""
                    - **Sleep Duration & Stress**: Short sleep duration (< 6.5 hours) combined with higher stress scores (>= 7) is the dominant signature for Insomnia.
                    - **Blood Pressure & BMI**: Blood pressure readings above 130/85 mmHg combined with Overweight or Obese BMI are strong indicators of Sleep Apnea in the dataset.
                    - **Model Certainty**: Probabilities reflect statistical similarity to patterns present in the training data, not a direct measurement of clinical symptoms.
                    """)

            except Exception as e:
                st.error(f"Error during prediction execution: {e}")

        # Persistent Mandatory Disclaimer
        st.markdown(f"<div class='disclaimer-box'><b>Disclaimer:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 3: DATASET INSIGHTS
# ==========================================
elif page == "Dataset Insights":
    st.markdown("<div class='main-header'>Dataset Insights & Exploratory Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Empirical distributions, statistical relationships, and clinical feature insights.</div>", unsafe_allow_html=True)

    df_sample = load_dataset_sample()
    if df_sample is not None:
        # High-level metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>Total Observations</div>
                <div class='metric-value'>374</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>Predictive Features</div>
                <div class='metric-value'>11</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>Disorder Prevalence</div>
                <div class='metric-value'>41.4%</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-title'>Healthy Subjects</div>
                <div class='metric-value'>58.6%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Interactive Visualizations")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Class & BMI Distributions",
            "Sleep Duration & Quality",
            "Stress & Health Factors",
            "Correlation Heatmap",
            "Raw Data Sample"
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.image("images/generated_analysis_plots/01_target_distribution.png", caption="Target Class Distribution: None (219), Sleep Apnea (78), Insomnia (77)", use_container_width=True)
            with c2:
                st.image("images/generated_analysis_plots/07_bmi_category_distribution.png", caption="BMI Category Distribution: Normal (216), Overweight (148), Obese (10)", use_container_width=True)
            st.image("images/generated_analysis_plots/11_bmi_category_vs_disorder.png", caption="Sleep Disorder Proportions Across BMI Groups (Overweight individuals suffer disproportionately from Sleep Apnea and Insomnia)", use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.image("images/generated_analysis_plots/03_sleep_duration_distribution.png", caption="Sleep Duration Distribution (Mean: 7.13 hrs)", use_container_width=True)
            with c2:
                st.image("images/generated_analysis_plots/08_sleep_duration_vs_disorder.png", caption="Sleep Duration vs Disorder Status", use_container_width=True)
            st.image("images/generated_analysis_plots/10_sleep_quality_vs_disorder.png", caption="Subjective Sleep Quality vs Disorder Category", use_container_width=True)

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.image("images/generated_analysis_plots/05_stress_level_distribution.png", caption="Stress Level Distribution (Scale 1-10)", use_container_width=True)
            with c2:
                st.image("images/generated_analysis_plots/09_stress_level_vs_disorder.png", caption="Stress Level by Sleep Disorder Status", use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                st.image("images/generated_analysis_plots/02_age_distribution.png", caption="Age Distribution of Participants (Median: 43 years)", use_container_width=True)
            with c4:
                st.image("images/generated_analysis_plots/06_physical_activity_distribution.png", caption="Physical Activity Level (Mean: 59.2 min/day)", use_container_width=True)

        with tab4:
            st.image("images/generated_analysis_plots/12_correlation_heatmap.png", caption="Pearson Correlation Heatmap Across Numerical Physiological Attributes", use_container_width=True)
            st.markdown("""
            **Key Correlation Findings:**
            - **Strong Positive Correlation (0.88)** between Sleep Duration and Quality of Sleep.
            - **Strong Negative Correlation (-0.81)** between Quality of Sleep and Stress Level.
            - **Strong Negative Correlation (-0.81)** between Sleep Duration and Stress Level.
            - **Substantial Positive Correlation (0.97)** between Systolic and Diastolic Blood Pressure.
            """)

        with tab5:
            st.markdown("#### Sample Records from Dataset (Excluding Personal Identifiers)")
            view_cols = [c for c in df_sample.columns if c != "Person ID"]
            st.dataframe(df_sample[view_cols].head(25), use_container_width=True)

    else:
        st.warning("Dataset not available for exploration.")

    st.markdown(f"<div class='disclaimer-box'><b>Disclaimer:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================
elif page == "Model Performance":
    st.markdown("<div class='main-header'>Model Evaluation & Benchmarking</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Rigorous comparison across 5 supervised classification algorithms using Stratified K-Fold CV.</div>", unsafe_allow_html=True)

    metrics_data, comparison_df = load_metrics()

    if metrics_data and comparison_df is not None:
        best_name = metrics_data["best_model"]
        summary = metrics_data["summary_metrics"]

        # Top Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Selected Champion</div>
                <div class='metric-value'>{best_name}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Test Set Accuracy</div>
                <div class='metric-value'>{summary['Test Accuracy'] * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>Test Macro F1-Score</div>
                <div class='metric-value'>{summary['Test Macro F1'] * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>5-Fold CV Macro F1</div>
                <div class='metric-value'>{summary['CV Macro F1'] * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Algorithm Comparison Table
        st.subheader("1. Algorithm Comparison Table")
        st.caption("All models were trained on the identical 80% stratified split with identical cross-validation folds.")
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=["Test Accuracy", "Test Macro F1", "CV Macro F1"], color="#D1FAE5"), use_container_width=True)

        # Performance Plots
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("2. Model Comparison Bar Chart")
            if os.path.exists("images/generated_analysis_plots/model_comparison.png"):
                st.image("images/generated_analysis_plots/model_comparison.png", use_container_width=True)
            else:
                st.write("Chart pending.")

        with col2:
            st.subheader(f"3. Confusion Matrix: {best_name}")
            if os.path.exists("images/generated_analysis_plots/confusion_matrix.png"):
                st.image("images/generated_analysis_plots/confusion_matrix.png", use_container_width=True)
            else:
                st.write("Confusion matrix plot pending.")

        st.markdown("---")
        # Per-class Classification Report Table
        st.subheader("4. Detailed Per-Class Evaluation Metrics (Test Set)")
        rep = metrics_data["classification_report"]
        
        per_class_rows = []
        for cls in ["Insomnia", "None", "Sleep Apnea"]:
            if cls in rep:
                per_class_rows.append({
                    "Class": cls,
                    "Precision": f"{rep[cls]['precision']:.4f}",
                    "Recall": f"{rep[cls]['recall']:.4f}",
                    "F1-Score": f"{rep[cls]['f1-score']:.4f}",
                    "Test Support (Samples)": int(rep[cls]['support']),
                })
        st.table(pd.DataFrame(per_class_rows))

        st.markdown("""
        #### Evaluation Insights:
        - **Healthy Class Identification**: The model achieves **100% Precision and 100% Recall** on the 'None' class, correctly differentiating healthy subjects without false positives.
        - **Sleep Apnea Performance**: Achieves **93.8% Recall** and **88.2% Precision**, reliably detecting individuals at risk of sleep-disordered breathing.
        - **Insomnia Performance**: Reaches **86.7% Recall** and **92.9% Precision**.
        - **Prioritizing Macro F1**: Because sleep disorders are medically significant and class distribution is moderately imbalanced (58.6% None, 20.9% Sleep Apnea, 20.6% Insomnia), **Macro F1-Score** was chosen as the primary selection criterion over raw accuracy alone.
        """)

    else:
        st.warning("Performance metrics data is currently unavailable. Run `python train_model.py` to generate evaluation artifacts.")

    st.markdown(f"<div class='disclaimer-box'><b>Disclaimer:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==========================================
# PAGE 5: ABOUT PROJECT
# ==========================================
elif page == "About Project":
    st.markdown("<div class='main-header'>About This Project</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>System architecture, machine learning methodology, and project documentation.</div>", unsafe_allow_html=True)

    st.markdown("""
    ### Project Architecture & Workflow
    ```mermaid
    flowchart LR
        A[Raw Input Data] --> B[Feature Engineering]
        B -->|Systolic/Diastolic BP & Clean BMI| C[ColumnTransformer]
        C -->|StandardScaler| D[Numerical Pipeline]
        C -->|OneHotEncoder| E[Categorical Pipeline]
        D & E --> F[Trained Random Forest Classifier]
        F --> G[Multi-Class Prediction: None / Insomnia / Sleep Apnea]
    ```
    
    ### Engineering Highlights
    1. **Data Leakage Prevention**:
       - Transformers and imputers are strictly fitted on the training split only (`fit_transform`) and applied out-of-sample (`transform`) on test and inference data.
    2. **Domain-Specific Preprocessing**:
       - Standardized ambiguous BMI entries (`Normal Weight` -> `Normal`).
       - Split compound string Blood Pressure into physiologically meaningful continuous attributes (`Systolic_BP` and `Diastolic_BP`).
    3. **Stratified Validation**:
       - Used 5-Fold Stratified Cross-Validation on training data to ensure proportional representation of minority disorder classes during model selection.
    4. **Inference Consistency**:
       - The deployed Streamlit web application utilizes the identical saved `sleep_disorder_pipeline.joblib` artifact, ensuring 100% training-inference parity.

    ### Limitations
    - **Dataset Size**: The Kaggle Sleep Health and Lifestyle dataset contains 374 observations. While high accuracy is achieved on the test set, larger and more diverse clinical cohorts are required for broad generalizability.
    - **Self-Reported Variables**: Sleep Quality and Stress Level are subjective ratings (1-10) which introduce participant subjectivity.
    - **Cross-Sectional Nature**: The data captures single-point observations rather than longitudinal sleep study polysomnography recordings.

    ### Technology Stack
    - **Language**: Python 3.14 / 3.11
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-Learn
    - **Model Persistence**: Joblib
    - **Visualization**: Matplotlib, Seaborn
    - **Web Framework**: Streamlit
    - **Version Control & Hosting**: Git, GitHub, Streamlit Community Cloud
    """)

    st.markdown(f"<div class='disclaimer-box'><b>Responsible Use & Educational Notice:</b><br>{MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)
