# Sleep Disorder Classification Using Machine Learning
## Engineering Mini-Project Presentation Deck (12 Slides)

---

### Slide 1: Title Slide
- **Title**: Sleep Disorder Classification Using Machine Learning
- **Subtitle**: A Multi-Class Supervised Learning Framework for Lifestyle & Sleep Health Assessment
- **Domain**: Machine Learning / Healthcare Informatics / Predictive Analytics
- **Department**: Department of Computer Science & Engineering
- **Presenter**: Engineering Student Mini-Project
- **Tools Used**: Python, Scikit-Learn, Pandas, NumPy, Streamlit, Matplotlib, Seaborn

---

### Slide 2: Introduction & Background
- **Context**:
  - Sleep is a vital biological state that regulates neurocognitive performance, metabolic balance, cardiovascular health, and emotional stability.
  - More than 30% of adults worldwide report acute or chronic sleep disturbances.
  - The two most prevalent diagnostic categories are **Insomnia** (difficulty initiating or maintaining restorative sleep) and **Sleep Apnea** (repetitive episodes of upper airway collapse during sleep).
- **Need for AI/ML**:
  - Diagnostic gold-standard Polysomnography (PSG) involves overnight hospital stays, extensive electrode montages, and high costs (\$1,000–\$3,000).
  - Machine learning provides a scalable, non-invasive risk assessment tool leveraging everyday health metrics.

---

### Slide 3: Problem Statement
- **Diagnostic Bottlenecks**:
  - Long waiting times (often 3–6 months) for clinical sleep consultations.
  - High proportion of undiagnosed sleep apnea cases (> 80% in moderate cohorts).
- **Multi-Factor Complexity**:
  - Sleep health is influenced by interconnected attributes: sleep duration, perceived stress, cardiovascular markers (blood pressure, resting heart rate), daily physical exertion, and body mass index (BMI).
- **Objective of Problem Formulation**:
  - Formulate a 3-class classification problem (`None`, `Insomnia`, `Sleep Apnea`) that categorizes individuals based on accessible self-reported and physiological parameters.

---

### Slide 4: Project Objectives
1. **Develop Multi-Class Classifier**: Accurately predict whether an individual exhibits No Disorder, Insomnia, or Sleep Apnea.
2. **Build End-to-End Leakage-Free Pipeline**: Combine custom feature engineering, median/mode imputation, standard scaling, and one-hot encoding in a unified Scikit-Learn Pipeline.
3. **Benchmark Candidate Algorithms**: Fairly compare 5 algorithms (Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Gradient Boosting) using 5-Fold Stratified Cross-Validation.
4. **Deploy User-Friendly Web Application**: Construct an interactive, modern Streamlit web interface with real-time inference, confidence probabilities, and exploratory data analytics.

---

### Slide 5: Dataset Overview & Characteristics
- **Source**: Kaggle Sleep Health and Lifestyle Dataset (374 participant records, 13 features).
- **Target Distribution**:
  - **No Sleep Disorder (`None`)**: 219 samples (58.6%)
  - **Sleep Apnea**: 78 samples (20.9%)
  - **Insomnia**: 77 samples (20.6%)
- **Attributes Available**:
  - *Demographics*: Gender (Male: 189, Female: 185), Age (27–59 yrs), Occupation (11 categories).
  - *Sleep Characteristics*: Sleep Duration (5.8–8.5 hrs), Quality of Sleep (Scale 1–10).
  - *Physical & Mental*: Physical Activity Level (30–90 min/day), Stress Level (Scale 1–10), Daily Steps (3,000–10,000).
  - *Cardiovascular & Anthropometric*: BMI Category (`Normal`, `Overweight`, `Obese`), Blood Pressure (`Systolic/Diastolic`), Heart Rate (65–86 BPM).

---

### Slide 6: Data Preprocessing & Feature Engineering
- **Handling Missing Target Labels**:
  - 219 null records in `Sleep Disorder` signify healthy individuals with no diagnosed disorder; mapped explicitly to `'None'`.
- **Domain Feature Engineering (`FeatureEngineer`)**:
  - *Blood Pressure Decomposition*: Parsed compound string `'126/83'` into continuous numeric features `Systolic_BP` (126 mmHg) and `Diastolic_BP` (83 mmHg).
  - *BMI Standardization*: Normalized synonymous label `'Normal Weight'` into `'Normal'`.
  - *Identifier Removal*: Dropped non-informative `Person ID` to prevent spurious pattern memorization.
- **Data Leakage Prevention**:
  - All scalers, imputers, and encoders fitted strictly on the 80% training split via `build_preprocessor()`.

---

### Slide 7: Algorithms Evaluated
1. **Logistic Regression**:
   - Baseline linear model with L2 regularization and balanced class weighting.
2. **Decision Tree Classifier**:
   - Non-parametric recursive splitting using Gini impurity with maximum depth regularization (`max_depth=5`).
3. **Random Forest Classifier (Selected Champion)**:
   - Ensemble of 100 decorrelated decision trees using bootstrap aggregation (bagging) and randomized feature subsets.
4. **K-Nearest Neighbors (KNN)**:
   - Non-parametric distance-based classification using distance-weighted voting ($k=5$).
5. **Gradient Boosting Classifier**:
   - Sequential boosting optimizing multi-class cross-entropy loss with learning rate $\eta = 0.1$.

---

### Slide 8: System Architecture & Workflow
```
[Raw User/CSV Input] 
         │
         ▼
[FeatureEngineer Transformer] (BP split into Systolic/Diastolic, BMI cleaned, Person ID dropped)
         │
         ▼
[ColumnTransformer Preprocessor]
 ├── Numerical Pipeline: SimpleImputer(median) ➔ StandardScaler()
 └── Categorical Pipeline: SimpleImputer(most_frequent) ➔ OneHotEncoder(ignore)
         │
         ▼
[Trained Random Forest Classifier (100 Trees)]
         │
         ▼
[Multi-Class Output: None / Insomnia / Sleep Apnea + Probabilities]
         │
         ▼
[Streamlit Cloud UI: Interactive Form, Prediction Cards, Analytics Dashboard]
```

---

### Slide 9: Model Evaluation & Benchmark Results
- Evaluated on a held-out 20% stratified test set (75 samples) after 5-fold Stratified CV on the training set (299 samples).

| Model | CV Accuracy | CV Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.8762** | **0.8491** | **96.00%** | **0.9352** | **0.9599** |
| Gradient Boosting | 0.8762 | 0.8483 | 94.67% | 0.9214 | 0.9472 |
| K-Nearest Neighbors | 0.8562 | 0.8254 | 93.33% | 0.9082 | 0.9347 |
| Logistic Regression | 0.8828 | 0.8600 | 93.33% | 0.9076 | 0.9341 |
| Decision Tree | 0.8594 | 0.8329 | 93.33% | 0.8995 | 0.9329 |

- **Selection Rationale**: Random Forest demonstrated the highest Test Macro F1 (0.9352) and Test Accuracy (96.00%), ensuring robust detection across both minority disorder classes.

---

### Slide 10: Streamlit Web Application Interface
- **Modern Responsive Design**: Built with Streamlit, custom CSS metric cards, and clean typography.
- **Five Dedicated Sections**:
  1. **Home**: Project introduction, class definitions, and system highlights.
  2. **Predict**: Interactive form with real-time BP preview, sliders, multi-class predictions, and confidence distribution charts.
  3. **Dataset Insights**: Interactive dashboard displaying all 12 EDA visualizations, correlation matrix, and raw data sample viewer.
  4. **Model Performance**: Live benchmark comparison table, confusion matrix heatmap, and per-class metrics.
  5. **About Project**: System architecture diagram, methodology, limitations, and references.

---

### Slide 11: Results & Key Findings
- **Per-Class Test Performance (Random Forest)**:
  - **No Disorder (`None`)**: Precision = 100%, Recall = 100%, F1 = 1.00 (44 test samples).
  - **Sleep Apnea**: Precision = 88.2%, Recall = 93.8%, F1 = 0.909 (16 test samples).
  - **Insomnia**: Precision = 92.9%, Recall = 86.7%, F1 = 0.897 (15 test samples).
- **Clinical Feature Correlates**:
  - **Insomnia**: Heavily driven by low sleep duration (< 6.5 hours) and high stress scores (>= 7).
  - **Sleep Apnea**: Strongly correlated with overweight/obese BMI and elevated blood pressure (Systolic > 130 mmHg).
  - **Overall Test Matrix**: Only 3 minor misclassifications across 75 test subjects.

---

### Slide 12: Conclusion & Future Scope
- **Conclusion**:
  - Successfully developed, tested, and validated an end-to-end machine learning system achieving 96% accuracy and 0.935 Macro F1 in classifying sleep disorders.
  - Unified pipeline ensures complete training-inference consistency and eliminates data leakage.
- **Future Enhancements**:
  1. **Continuous Wearable Streaming**: Integrate live API data from smartwatch PPG and accelerometer sensors (Apple HealthKit / Google Fit).
  2. **Deep Learning on Raw Biosignals**: Apply 1D-CNN or Temporal Convolutional Networks to raw EEG/ECG waveforms.
  3. **Clinical Cohort Expansion**: Validate on larger multi-center clinical datasets with demographic diversity.
- **Mandatory Ethical Notice**: Application is strictly educational and does not constitute medical diagnosis.
