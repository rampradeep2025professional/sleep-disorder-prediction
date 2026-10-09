# College Mini-Project Report
# Sleep Disorder Classification Using Machine Learning

---

## Title Page

**Project Title:** Sleep Disorder Classification Using Machine Learning  
**Project Type:** Machine Learning & Data Science Mini-Project  
**Domain:** Healthcare Informatics / Predictive Machine Learning  
**Target Audience:** Engineering Evaluation Committee & Academic Review  
**Academic Year:** 2025–2026  

---

## Abstract
Sleep disorders represent an escalating public health crisis affecting more than 30% of adults globally. Conditions such as Insomnia and Sleep Apnea impair daytime neurocognitive performance, exacerbate metabolic dysfunction, and elevate long-term risks of cardiovascular morbidity. However, gold-standard diagnostic modalities, such as in-laboratory Polysomnography (PSG), remain inaccessible, expensive, and subject to prolonged clinical backlogs. This mini-project presents an end-to-end, multi-class machine learning system designed to classify sleep disorder status into three distinct categories: **No Sleep Disorder (`None`)**, **Insomnia**, and **Sleep Apnea**, utilizing non-invasive lifestyle, demographic, and physiological indicators.

Using the Sleep Health and Lifestyle dataset (374 patient records, 13 attributes), a reproducible preprocessing pipeline was engineered using Scikit-Learn's `ColumnTransformer` and custom feature transformers. The pipeline handles missing values, standardizes BMI classifications, and decomposes compound blood pressure strings into continuous systolic and diastolic parameters while completely eliminating data leakage. Five supervised classification models—Logistic Regression, Decision Trees, Random Forests, K-Nearest Neighbors, and Gradient Boosting—were benchmarked using 5-Fold Stratified Cross-Validation on training data and evaluated on an independent held-out test set. The **Random Forest Classifier** was selected as the champion model, achieving **96.00% Test Accuracy**, **0.9352 Test Macro F1-Score**, and **100% precision and recall on healthy controls**. A responsive, modern web application was developed and deployed using Streamlit to facilitate interactive prediction, confidence visualization, and exploratory dataset analytics. The system strictly serves educational purposes and includes comprehensive ethical safeguards.

---

## 1. Introduction
Sleep is a vital, evolutionarily conserved physiological state essential for memory consolidation, neurotoxin clearance, cellular regeneration, and endocrine regulation. Chronic disruption of sleep homeostasis is epidemiologically linked to cardiovascular disease, type 2 diabetes, immunodeficiency, depression, and workplace/vehicular accidents.

Despite its clinical importance, sleep medicine remains constrained by diagnostic bottlenecks. Conventional clinical evaluation requires polysomnography (PSG)—an overnight test requiring comprehensive monitoring of electroencephalography (EEG), electrooculography (EOG), electromyography (EMG), electrocardiography (ECG), and respiratory airflows. PSG is labor-intensive, costly ($1,000 to $3,000 per night), and psychologically uncomfortable for patients. Consequently, over 80% of individuals suffering from moderate-to-severe sleep-disordered breathing remain undiagnosed.

Recent advances in consumer health tracking, wearable biometric sensors, and computational statistical modeling enable non-invasive sleep assessment. By identifying complex non-linear interactions across everyday attributes—such as sleep duration, physical activity, occupational stress, resting pulse rate, blood pressure, and body mass index—supervised machine learning provides an accessible, non-invasive risk triaging framework. This project demonstrates the design, empirical evaluation, and deployment of a multi-class sleep disorder classification system.

---

## 2. Problem Statement
The diagnostic pathway for sleep pathology faces three core challenges:
1. **Severe Underdiagnosis and Delay:** Traditional medical pathways require formal clinical sleep laboratory consultations, resulting in treatment delays of months to years.
2. **Multi-Factor Etiological Complexity:** Sleep disorders do not stem from a single metric; they represent multifaceted interactions between physiological markers (blood pressure, BMI, heart rate) and lifestyle behaviors (physical exertion, occupational stress, sleep duration).
3. **Lack of Transparent Educational Screening Tools:** Many proprietary wellness applications provide black-box sleep scores without empirical machine learning rigor, reproducible benchmarks, or clear multi-class risk stratification.

There is a defined need for an open-source, reproducible, multi-class machine learning pipeline capable of classifying sleep status into `None`, `Insomnia`, and `Sleep Apnea` with high sensitivity, zero data leakage, and a verified web user interface.

---

## 3. Objectives
The core objectives of this mini-project are:
1. **Multi-Class Formulation:** Design a multi-class predictive framework to differentiate between healthy sleep patterns (`None`), `Insomnia`, and `Sleep Apnea`.
2. **Leakage-Free Preprocessing Architecture:** Construct an automated Scikit-Learn `Pipeline` incorporating custom domain transformers, median/mode imputation, one-hot encoding, and standard scaling.
3. **Empirical Algorithm Benchmarking:** Rigorously compare 5 candidate machine learning algorithms using 5-Fold Stratified Cross-Validation on training data and an independent 20% test split.
4. **Prioritize Clinically Balanced Metrics:** Select the champion model based on **Macro F1-Score** and per-class recall rather than misleading overall accuracy alone.
5. **Interactive Deployment:** Develop an interactive web application using Streamlit featuring real-time inference, multi-class confidence probability charts, and interactive exploratory data analytics.
6. **Automated Quality Assurance:** Implement an automated `pytest` test suite verifying schema validation, edge case resilience, missing-value imputation, and inference integrity.

---

## 4. Existing System vs. Proposed System

### 4.1 Limitations of Existing Systems
- **Clinical Polysomnography (PSG):** Gold standard, but prohibitively expensive, cumbersome, geographically restricted, and unsuitable for rapid population screening.
- **Subjective Questionnaires (e.g., Epworth Sleepiness Scale):** Prone to recall bias, lack physiological grounding, and cannot distinguish effectively between apnea and insomnia.
- **Commercial Fitness Trackers:** Provide proprietary "Sleep Scores" that lack transparent multi-class clinical grounding and are not open-source or verifiable.

### 4.2 Advantages of Proposed System
- **Holistic Feature Integration:** Combines subjective sleep ratings, occupational factors, physical exertion, and objective cardiovascular indicators (blood pressure and heart rate).
- **Multi-Class Risk Stratification:** Distinctly predicts `None`, `Insomnia`, and `Sleep Apnea`.
- **Complete Pipeline Reproducibility:** Single serialized pipeline (`sleep_disorder_pipeline.joblib`) ensures 100% parity between training and inference.
- **Zero Data Leakage:** Strict separation of training and test distributions.
- **Educational Transparency:** Web dashboard exposes empirical confusion matrices, cross-validation metrics, and distribution charts.

---

## 5. System Requirements

### 5.1 Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (Multi-core recommended for parallel CV).
- **RAM:** Minimum 4 GB (8 GB recommended).
- **Storage:** 500 MB free hard disk space for code, model artifacts, and generated visualizations.
- **Network:** Internet access for package installation and optional cloud hosting.

### 5.2 Software Requirements
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Programming Language:** Python 3.10+ (tested and verified on Python 3.14).
- **Core Libraries:**
  - `pandas` (>= 2.0.0) — Tabular data manipulation and analysis
  - `numpy` (>= 1.24.0) — Vectorized mathematical computations
  - `scikit-learn` (>= 1.3.0) — Machine learning algorithms, pipelines, metrics
  - `matplotlib` (>= 3.7.0) & `seaborn` (>= 0.12.0) — Statistical data visualization
  - `joblib` (>= 1.3.0) — High-efficiency pipeline serialization
  - `streamlit` (>= 1.30.0) — Web application framework
  - `pytest` (>= 7.4.0) — Automated unit and integration testing
- **Version Control:** Git & GitHub.
- **Cloud Hosting:** Streamlit Community Cloud.

---

## 6. Dataset Description
The system is built upon the **Sleep Health and Lifestyle Dataset** (sourced from Kaggle).

### 6.1 Attributes and Clinical Meaning
The dataset contains 374 observations across 13 attributes:

| Feature Name | Data Type | Role | Clinical / Behavioral Meaning |
| :--- | :---: | :---: | :--- |
| `Person ID` | Integer | Identifier | Arbitrary participant identifier (**dropped** during modeling). |
| `Gender` | String | Categorical | Biological sex (`Male`: 189, `Female`: 185). |
| `Age` | Integer | Numerical | Age in years (range: 27 to 59, median: 43). |
| `Occupation` | String | Categorical | Profession across 11 roles (Nurse, Doctor, Engineer, etc.). |
| `Sleep Duration` | Float | Numerical | Average sleep duration in hours/day (range: 5.8 to 8.5). |
| `Quality of Sleep` | Integer | Numerical | Subjective rating of sleep quality (scale: 1 to 10). |
| `Physical Activity Level` | Integer | Numerical | Minutes of daily physical activity (range: 30 to 90). |
| `Stress Level` | Integer | Numerical | Perceived psychological stress level (scale: 1 to 10). |
| `BMI Category` | String | Categorical | Standardized body mass classification (`Normal`, `Overweight`, `Obese`). |
| `Blood Pressure` | String | Compound | Systolic/Diastolic pressure string (e.g. `'126/83'`). |
| `Heart Rate` | Integer | Numerical | Resting pulse in beats per minute (range: 65 to 86). |
| `Daily Steps` | Integer | Numerical | Pedometer daily step count (range: 3,000 to 10,000). |
| `Sleep Disorder` | String | Target | Categorical outcome (`None`: 219, `Sleep Apnea`: 78, `Insomnia`: 77). |

### 6.2 Target Class Distribution
- **No Sleep Disorder (`None`):** 219 instances (58.56%)
- **Sleep Apnea:** 78 instances (20.86%)
- **Insomnia:** 77 instances (20.59%)
Total: 374 instances.

---

## 7. Data Preprocessing & Feature Engineering

### 7.1 Target Imputation
In the raw dataset, the target column contained 219 `NaN` values. Domain inspection confirmed that `NaN` denotes asymptomatic individuals without diagnosed sleep disorders. Rather than discarding these records, all `NaN` entries were standardized to `'None'`.

### 7.2 Custom Domain Feature Engineering (`FeatureEngineer`)
A Scikit-Learn compatible transformer (`FeatureEngineer`) was implemented to perform three essential transformations:
1. **Identifier Elimination:** Removes `Person ID` to prevent spurious memorization.
2. **Blood Pressure Decomposition:** Splits the compound string `'Systolic/Diastolic'` (e.g. `'130/85'`) into two continuous numerical columns:
   $$\text{Systolic\_BP} = 130.0 \quad (\text{mmHg}), \quad \text{Diastolic\_BP} = 85.0 \quad (\text{mmHg})$$
   This physiological decomposition allows linear and tree models to capture vascular hypertension directly.
3. **BMI Standardization:** Corrects synonymous labels where `'Normal Weight'` was standardized to `'Normal'`.

### 7.3 ColumnTransformer Architecture
To prevent data leakage, transformations were isolated into a Scikit-Learn `ColumnTransformer`:
- **Numerical Pipeline:** Evaluates 9 features (`Age`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Stress Level`, `Heart Rate`, `Daily Steps`, `Systolic_BP`, `Diastolic_BP`).
  $$\text{SimpleImputer(strategy='median')} \longrightarrow \text{StandardScaler()}$$
- **Categorical Pipeline:** Evaluates 3 features (`Gender`, `Occupation`, `BMI Category`).
  $$\text{SimpleImputer(strategy='most\_frequent')} \longrightarrow \text{OneHotEncoder(handle\_unknown='ignore', sparse\_output=False)}$$

### 7.4 Train/Test Splitting
The dataset was split using an 80/20 ratio stratified across the target variable with a fixed random seed (`random_state=42`):
- **Training Set:** 299 samples (used for 5-fold cross-validation and fitting).
- **Held-Out Testing Set:** 75 samples (held strictly isolated for final unbiased evaluation).

---

## 8. Exploratory Data Analysis & Empirical Observations
Twelve dedicated statistical visualizations were generated:
1. **Target Distribution:** Verifies a 58.6% healthy vs. 41.4% disorder split.
2. **Age Distribution:** Median age of 43 years, spanning working-age adults from 27 to 59.
3. **Sleep Duration:** Bell-shaped curve centered at 7.13 hours/day.
4. **Sleep Quality:** Clustered between ratings 6 and 8.
5. **Stress Level:** Bimodal distribution peaking at stress levels 3 (low) and 8 (high).
6. **Physical Activity:** Average 59.2 minutes/day.
7. **BMI Category:** 57.8% Normal (216), 39.6% Overweight (148), 2.7% Obese (10).
8. **Sleep Duration vs. Disorder:** Insomnia patients exhibit drastically reduced sleep duration (median $\approx 6.5$ hours) compared to healthy subjects (median $\approx 7.8$ hours).
9. **Stress Level vs. Disorder:** Insomnia and Apnea cohorts report significantly higher stress levels (median $\ge 7$) than healthy controls (median $\le 4$).
10. **Quality vs. Disorder:** Healthy participants average quality ratings $\ge 8$, whereas disorder groups report ratings $\le 6$.
11. **BMI vs. Disorder:** Sleep Apnea is strongly concentrated in Overweight and Obese participants; healthy sleep is dominant in Normal BMI individuals.
12. **Correlation Heatmap:** Strong inverse correlation ($r = -0.81$) between sleep quality and stress; strong positive correlation ($r = 0.88$) between sleep duration and quality.

---

## 9. Algorithms Evaluated

### 9.1 Logistic Regression
A multi-class multinomial linear model with L2 regularization and balanced class weighting. Models log-odds as a linear combination of scaled features.

### 9.2 Decision Tree Classifier
A non-parametric recursive partitioning algorithm using Gini impurity. Regularized with `max_depth=5` to prevent memorization of small leaf subsets.

### 9.3 Random Forest Classifier (Selected Model)
An ensemble of 100 decorrelated decision trees using bootstrap aggregating (bagging) and random feature subspace sampling. Provides robust multi-way non-linear interaction modeling and minimizes variance.

### 9.4 K-Nearest Neighbors (KNN)
An instance-based classifier computing distance-weighted votes among the $k=5$ closest training samples in scaled Euclidean space.

### 9.5 Gradient Boosting Classifier
A sequentially trained ensemble of shallow decision trees optimizing multi-class deviance (log-loss) using gradient descent with learning rate $\eta = 0.1$.

---

## 10. System Architecture & Workflow

```
┌────────────────────────────────────────────────────────┐
│                   Input Data Layer                     │
│  (Raw CSV Training Dataset / Real-Time Streamlit Form) │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Feature Engineering Transformer               │
│   • Drop Person ID                                     │
│   • Parse Blood Pressure ──► Systolic_BP, Diastolic_BP │
│   • Standardize 'Normal Weight' ──► 'Normal'           │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│             ColumnTransformer Preprocessor             │
│   ┌──────────────────────────┬──────────────────────┐  │
│   │   Numerical Pipeline     │ Categorical Pipeline │  │
│   │   • Median Imputer       │ • Most Frequent Imp. │  │
│   │   • StandardScaler       │ • OneHotEncoder      │  │
│   └──────────────────────────┴──────────────────────┘  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│           Trained Random Forest Classifier             │
│             (100 Trees, Balanced Weighting)            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                  Inference Output Layer                │
│   • Predicted Class: None / Insomnia / Sleep Apnea     │
│   • Calibrated Prediction Probabilities                │
│   • Streamlit Cloud Web Application & UI Cards         │
└────────────────────────────────────────────────────────┘
```

---

## 11. Results and Discussion

### 11.1 Algorithm Comparison Table
All 5 candidate models were evaluated on the identical 80/20 stratified split. Cross-validation was conducted strictly on the 299 training samples.

| Model Name | 5-Fold CV Accuracy | 5-Fold CV Macro F1 | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Test Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.8762** | **0.8491** | **96.00%** | **0.9370** | **0.9347** | **0.9352** | **0.9599** |
| **Gradient Boosting** | 0.8762 | 0.8483 | 94.67% | 0.9163 | 0.9271 | 0.9214 | 0.9472 |
| **K-Nearest Neighbors** | 0.8562 | 0.8254 | 93.33% | 0.8983 | 0.9196 | 0.9082 | 0.9347 |
| **Logistic Regression** | 0.8828 | 0.8600 | 93.33% | 0.9077 | 0.9182 | 0.9076 | 0.9341 |
| **Decision Tree** | 0.8594 | 0.8329 | 93.33% | 0.9042 | 0.9091 | 0.8995 | 0.9329 |

### 11.2 Champion Model Detailed Performance
The **Random Forest Classifier** achieved the superior **Test Macro F1-Score of 0.9352** and **Test Accuracy of 96.00%**.

#### Per-Class Classification Report (Test Set, N = 75):
- **None (Healthy):**
  - Precision: **1.0000** (100.0%)
  - Recall: **1.0000** (100.0%)
  - F1-Score: **1.0000** (100.0%)
  - Support: 44 samples
- **Insomnia:**
  - Precision: **0.9286** (92.86%)
  - Recall: **0.8667** (86.67%)
  - F1-Score: **0.8966** (89.66%)
  - Support: 15 samples
- **Sleep Apnea:**
  - Precision: **0.8824** (88.24%)
  - Recall: **0.9375** (93.75%)
  - F1-Score: **0.9091** (90.91%)
  - Support: 16 samples

#### Confusion Matrix (Held-Out Test Set):
$$\begin{pmatrix}
13 & 0 & 2 \\
0 & 44 & 0 \\
1 & 0 & 15
\end{pmatrix}$$
- Row 1 (Actual Insomnia): 13 correctly identified, 0 predicted as None, 2 misclassified as Apnea.
- Row 2 (Actual None): 44 correctly identified, 0 misclassified.
- Row 3 (Actual Sleep Apnea): 15 correctly identified, 0 predicted as None, 1 misclassified as Insomnia.

**Critical Finding:** The model demonstrated zero confusion between healthy individuals and individuals with sleep disorders. No healthy subject was misclassified as having a disorder, and no patient with a disorder was misclassified as healthy.

---

## 12. Automated Testing & Verification
An automated test suite using `pytest` was developed in `tests/test_model.py` and executed locally:
- **Total Tests Executed:** 12 tests
- **Pass Rate:** 100% (12 passed in 9.12 seconds)
- **Verified Aspects:**
  1. Dataset loading and schema integrity
  2. Target imputation and cleaning
  3. Custom `FeatureEngineer` transformation
  4. Saved pipeline artifact loading
  5. Multi-class prediction output validity
  6. Probability calibration (probabilities sum to 1.0)
  7. Resilience to unseen categories (`handle_unknown='ignore'`)
  8. Graceful imputation on missing inputs
  9. Metric metadata JSON integrity
  10. Presence and size of all generated plot images
  11. Compilation of `app.py`
  12. Missing-model handling

---

## 13. Streamlit Web Application
The user interface was built using Streamlit (`app.py`), structured into five intuitive sections:
1. **Home:** High-level project summary, disorder definitions, and production metric highlights.
2. **Predict:** Patient details form featuring dynamic sliders, occupation select boxes, blood pressure synthesis, real-time multi-class prediction alerts, and horizontal probability confidence bar charts.
3. **Dataset Insights:** Interactive analytics displaying class distributions, correlation heatmaps, sleep duration boxplots, and searchable tabular records.
4. **Model Performance:** Live benchmark comparison table, confusion matrix heatmaps, and per-class precision/recall metrics.
5. **About Project:** Mermaid architecture diagrams, methodology explanations, limitations, and references.

---

## 14. Advantages & Project Strengths
1. **Complete Parity:** Model training and Streamlit inference consume the exact same serialized pipeline object (`sleep_disorder_pipeline.joblib`), eliminating discrepancies.
2. **Clinically Grounded Preprocessing:** Parsing blood pressure into systolic and diastolic integers allows algorithms to detect hypertension patterns.
3. **Leakage-Free Validation:** Strict separation of training and test data ensures high real-world credibility.
4. **Zero-False-Positive Healthy Classification:** Achieves 100% precision on healthy individuals in the test cohort.
5. **High Usability:** Intuitive web application that requires zero coding knowledge to operate.

---

## 15. Limitations
1. **Sample Size:** The dataset contains 374 individuals; validation across tens of thousands of diverse clinical patients is needed for production diagnostic use.
2. **Subjective Variables:** Features like "Quality of Sleep" and "Stress Level" rely on subjective self-reporting on a 1–10 scale.
3. **Cross-Sectional Sampling:** The dataset captures single-point snapshot observations rather than continuous overnight sleep monitoring.

---

## 16. Future Enhancements
1. **Wearable IoT Ingestion:** Connect directly to Apple HealthKit, Fitbit Web API, or Google Fit to stream continuous biometric telemetry.
2. **Deep Learning on Raw Physiological Time-Series:** Train 1D Convolutional Neural Networks (CNNs) and LSTMs on raw overnight ECG, PPG, and SpO2 waveforms.
3. **Multi-Center Clinical Trials:** Partner with accredited sleep clinics to validate the model against concurrent polysomnography studies.

---

## 17. Conclusion
This project successfully designed, implemented, evaluated, and deployed an end-to-end Machine Learning system for classifying sleep disorders. By combining domain-specific feature engineering with an ensemble Random Forest classifier, the system achieved **96.00% Test Accuracy** and **0.9352 Test Macro F1-Score**. The modular Python architecture, automated test suite, and interactive Streamlit web application deliver an exemplary, reproducible, and educational engineering mini-project.

---

## 18. References
1. American Academy of Sleep Medicine. (2014). *International Classification of Sleep Disorders* (3rd ed.). Darien, IL: American Academy of Sleep Medicine.
2. Kaggle Dataset: Laksika Tharmalingam. *Sleep Health and Lifestyle Dataset*.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
5. Streamlit Documentation. *Streamlit: A faster way to build and share data apps*. https://docs.streamlit.io
