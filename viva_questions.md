# Viva Voce Questions & Answers
## Sleep Disorder Classification Using Machine Learning
### Comprehensive Mini-Project Examination Guide (30 Questions & Answers)

---

### Category 1: Foundations of Machine Learning & Classification

#### Q1: What type of machine learning problem does this project solve?
**Answer:** This project solves a **supervised, multi-class classification problem**.
- **Supervised:** The dataset contains labeled ground truth targets (`Sleep Disorder`).
- **Multi-Class Classification:** The target has three distinct mutually exclusive classes: `None` (No sleep disorder), `Insomnia`, and `Sleep Apnea`.

#### Q2: What is the fundamental difference between classification and regression?
**Answer:**
- **Classification** predicts discrete categorical labels (e.g., predicting whether a patient has `None`, `Insomnia`, or `Sleep Apnea`).
- **Regression** predicts continuous numerical quantities (e.g., predicting exact sleep duration in hours or blood pressure in mmHg).

#### Q3: Why is this project formulated as multi-class rather than binary classification?
**Answer:** Clinically, sleep pathology is not a binary "healthy vs. sick" spectrum. Insomnia (a disorder of sleep initiation/maintenance often related to psychological stress) and Sleep Apnea (a physiological respiratory disorder linked to airway obstruction, blood pressure, and BMI) have completely different etiologies and treatments. Classifying them into distinct categories provides much higher educational and triage value than binary detection.

---

### Category 2: Dataset, Features & Cleaning

#### Q4: Describe the dataset used in this project.
**Answer:** We used the **Sleep Health and Lifestyle Dataset** (available on Kaggle).
- **Total records:** 374 individuals.
- **Attributes:** 13 columns including demographic (Gender, Age, Occupation), lifestyle (Physical Activity, Daily Steps, Sleep Duration, Quality of Sleep, Stress Level), physiological (BMI Category, Blood Pressure, Heart Rate), and the target (`Sleep Disorder`).
- **Target Distribution:** None (219 samples, 58.6%), Sleep Apnea (78 samples, 20.9%), Insomnia (77 samples, 20.6%).

#### Q5: How were missing values in the dataset handled?
**Answer:** In the raw dataset, the `Sleep Disorder` column contained 219 `NaN` (missing) values. In domain context and Kaggle documentation, missing entries indicate healthy participants with **no diagnosed sleep disorder**. We explicitly imputed `NaN` as `'None'`. All other feature columns had zero missing values in the original 374 rows. In our pipeline, we also included `SimpleImputer` (median for numerical, most frequent for categorical) to ensure zero failures during inference if users provide incomplete inputs.

#### Q6: Why did you drop the `Person ID` column?
**Answer:** `Person ID` is an arbitrary sequence number (1 to 374). It possesses zero predictive relationship to human biology. If retained, models (especially tree-based algorithms) could memorize arbitrary ID ranges, causing catastrophic overfitting and target leakage.

#### Q7: How did you handle the `Blood Pressure` column?
**Answer:** The original dataset stored blood pressure as a compound text string (e.g., `"126/83"`). Instead of treating it as 25 distinct categorical text labels, we implemented a custom Scikit-Learn transformer (`FeatureEngineer`) that parsed the string into two continuous physiological numerical features:
1. `Systolic_BP` (e.g., 126.0 mmHg)
2. `Diastolic_BP` (e.g., 83.0 mmHg)
This preserves natural numerical ordering and clinical significance.

#### Q8: What anomaly was discovered in the `BMI Category` column and how was it corrected?
**Answer:** The dataset contained four labels: `Normal` (195), `Overweight` (148), `Normal Weight` (21), and `Obese` (10). `"Normal Weight"` and `"Normal"` represent the exact same clinical category ($18.5 \le \text{BMI} < 25.0$). We standardized `"Normal Weight"` to `"Normal"`, resulting in three clean categories: `Normal` (216), `Overweight` (148), and `Obese` (10).

#### Q9: How were categorical variables encoded?
**Answer:** We used **One-Hot Encoding** (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`) for `Gender`, `Occupation`, and `BMI Category`.
- `handle_unknown='ignore'` ensures that if an unseen category appears during inference (e.g., a new occupation or missing value), the pipeline sets all corresponding dummy columns to 0 rather than throwing a runtime error.

---

### Category 3: Preprocessing, Data Leakage & Pipelines

#### Q10: What is data leakage and how did you prevent it?
**Answer:** **Data leakage** occurs when information from outside the training dataset (such as the test set) influences model training, producing overly optimistic evaluation metrics that fail in real-world deployment.
We strictly prevented data leakage by:
1. Performing a stratified train/test split **before** any scaling or transformation.
2. Wrapping our transformations in Scikit-Learn's `ColumnTransformer` and `Pipeline`, which fits transformers exclusively on `X_train` (`fit_transform`) and only transforms `X_test` without refitting (`transform`).

#### Q11: Why is numerical feature scaling necessary?
**Answer:** Numerical features like `Daily Steps` (range 3,000–10,000) have vastly larger scales than `Sleep Duration` (5.8–8.5) or `Stress Level` (3–8). For distance-based algorithms like **K-Nearest Neighbors** and gradient-based models like **Logistic Regression**, unscaled features cause large-magnitude variables to dominate distance metrics. We used `StandardScaler` to transform features to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).

#### Q12: Why use a Scikit-Learn `Pipeline`?
**Answer:** A `Pipeline` chains sequential steps (custom feature engineering $\rightarrow$ ColumnTransformer $\rightarrow$ Classifier) into a single unified object.
Benefits:
- Eliminates code duplication between training and inference (`app.py`).
- Prevents train-test leakage.
- Enables saving and loading the complete artifact (`joblib.dump`) so the Streamlit application can accept raw patient inputs directly without manual preprocessing steps.

---

### Category 4: Model Training, Evaluation & Algorithms

#### Q13: What algorithms did you benchmark, and which model performed best?
**Answer:** We evaluated 5 candidate algorithms:
1. **Random Forest Classifier (Champion - Best Model)**: Test Accuracy: **96.00%**, Test Macro F1: **0.9352**, CV Macro F1: **0.8491**.
2. **Gradient Boosting Classifier**: Test Accuracy: 94.67%, Test Macro F1: 0.9214.
3. **K-Nearest Neighbors**: Test Accuracy: 93.33%, Test Macro F1: 0.9082.
4. **Logistic Regression**: Test Accuracy: 93.33%, Test Macro F1: 0.9076.
5. **Decision Tree**: Test Accuracy: 93.33%, Test Macro F1: 0.8995.

#### Q14: How does a Random Forest Classifier work?
**Answer:** Random Forest is an **ensemble learning** method based on **Bagging (Bootstrap Aggregating)**:
1. It builds an ensemble of $B$ decision trees ($B=100$ in our project).
2. Each tree is trained on a distinct bootstrap sample drawn with replacement from the training data.
3. At each split in a tree, only a random subset of features ($\sqrt{p}$) is considered.
4. During prediction, individual tree outputs are aggregated by majority voting. This decorrelates individual trees and dramatically reduces prediction variance without increasing bias.

#### Q15: Why did Random Forest outperform a single Decision Tree?
**Answer:** Single decision trees have high variance and are prone to overfitting training quirks. Random Forest averages predictions across 100 decorrelated trees, dampening variance, smoothing decision boundaries, and providing superior generalization on unseen data.

#### Q16: How does K-Nearest Neighbors (KNN) work?
**Answer:** KNN is an instance-based (lazy learning) algorithm. To classify a test instance $x$, KNN computes the Euclidean distance between $x$ and all training instances, identifies the $k$ closest neighbors ($k=5$), and assigns the class with the highest weighted vote among those neighbors.

#### Q17: What is Stratified K-Fold Cross-Validation?
**Answer:** In standard K-Fold CV, the dataset is split randomly into $K$ equal folds. In **Stratified K-Fold**, each fold preserves the exact class percentage ratio of the overall dataset (58.6% None, 20.9% Sleep Apnea, 20.6% Insomnia). This ensures minority classes are fairly represented in every validation fold, preventing biased performance estimates. We used $K=5$.

---

### Category 5: Metrics & Model Selection

#### Q18: What is the difference between Accuracy and Macro F1-Score?
**Answer:**
- **Accuracy** is the ratio of correct predictions to total predictions: $\frac{TP + TN}{TP + TN + FP + FN}$. On imbalanced datasets, accuracy can be deceptively high if the majority class is predicted correctly while minority classes fail.
- **Macro F1-Score** calculates the unweighted arithmetic mean of F1-scores across all individual classes: $\frac{F1_{\text{None}} + F1_{\text{Insomnia}} + F1_{\text{Apnea}}}{3}$. It gives equal importance to every class regardless of sample count, making it the superior metric for clinical risk detection.

#### Q19: Define Precision, Recall, and F1-Score.
**Answer:**
- **Precision:** $\frac{TP}{TP + FP}$ — Out of all patients predicted to have a disorder, what proportion actually had it? (Measures false alarm rate).
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — Out of all patients who genuinely had the disorder, what proportion was detected? (Measures missed diagnosis rate).
- **F1-Score:** The harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.

#### Q20: Explain the Confusion Matrix of your champion model.
**Answer:** Evaluated on the 75 held-out test samples:
- **None (Healthy)**: 44 actual samples $\rightarrow$ 44 correctly predicted as None (100% Precision, 100% Recall).
- **Insomnia**: 15 actual samples $\rightarrow$ 13 correctly predicted as Insomnia, 2 misclassified as Sleep Apnea.
- **Sleep Apnea**: 16 actual samples $\rightarrow$ 15 correctly predicted as Sleep Apnea, 1 misclassified as Insomnia.
Total test accuracy = $\frac{44 + 13 + 15}{75} = \frac{72}{75} = 96.00\%$.

#### Q21: How did you handle class imbalance in model training?
**Answer:**
1. Employed **stratified sampling** during train/test split and across all 5 cross-validation folds.
2. Configured `class_weight='balanced'` in candidate estimators (Logistic Regression, Decision Tree, Random Forest) to automatically adjust weights inversely proportional to class frequencies ($w_j = \frac{n}{k \cdot n_j}$).
3. Used **Macro F1-Score** as the primary selection criterion rather than raw accuracy.

---

### Category 6: Application, Deployment & Architecture

#### Q22: What is Streamlit and why was it chosen for this project?
**Answer:** Streamlit is an open-source Python framework designed for rapidly creating interactive data science and machine learning web applications. It executes purely in Python, provides native widgets (sliders, selects, metrics, charts), runs responsively in any browser, and integrates natively with Pandas and Scikit-Learn.

#### Q23: How does `app.py` load and use the trained model?
**Answer:** Instead of retraining the model on every page load, `app.py` uses Streamlit's `@st.cache_resource` decorator to load the pre-trained `model/sleep_disorder_pipeline.joblib` once into server memory. When the user submits the input form, the app packages inputs into a one-row Pandas DataFrame and directly calls `pipeline.predict()` and `pipeline.predict_proba()`.

#### Q24: What does the prediction probability chart in the app represent?
**Answer:** It visualizes the calibrated class posterior probabilities estimated by the Random Forest classifier (the proportion of individual trees that voted for each class). It represents statistical pattern similarity to training records, **not** a clinical probability of medical diagnosis.

#### Q25: How is this application deployed to Streamlit Community Cloud?
**Answer:**
1. The project repository is pushed to GitHub with `app.py`, `requirements.txt`, and `model/sleep_disorder_pipeline.joblib`.
2. On `share.streamlit.io`, the repository, branch (`main`), and main file (`app.py`) are specified.
3. Streamlit Cloud provisions a container, runs `pip install -r requirements.txt`, executes `streamlit run app.py`, and publishes a secure, public HTTPS URL.

---

### Category 7: Ethics, Limitations & Future Work

#### Q26: Why is an educational disclaimer mandatory in this application?
**Answer:** Machine learning models trained on survey datasets are statistical pattern matching systems, not certified medical diagnostic devices. Presenting model predictions as medical diagnoses could lead to self-medication, anxiety, or neglected clinical care. Our app prominently features an educational disclaimer instructing users to consult licensed healthcare providers.

#### Q27: What are the primary limitations of the current system?
**Answer:**
1. **Dataset Size:** 374 samples is relatively small for deep generalization across diverse global populations.
2. **Subjective Measures:** Sleep Quality and Stress Level are self-reported scores (1–10) that vary by personal perception.
3. **Cross-Sectional Data:** The data represents static snapshot observations rather than continuous overnight polysomnography (PSG) recordings.

#### Q28: How could this project be improved in the future?
**Answer:**
1. **IoT / Wearable Streaming:** Ingesting real-time PPG, heart rate variability (HRV), and 3D-accelerometer data from smartwatches (Apple Watch, Garmin, Fitbit).
2. **Deep Learning on Raw Signals:** Using 1D Convolutional Neural Networks (CNNs) or LSTMs on raw overnight EEG, ECG, and SpO2 time-series data.
3. **Multi-Center Clinical Cohorts:** Validating the model on diverse clinical datasets from multiple sleep research laboratories.

#### Q29: What is the significance of the Blood Pressure and BMI correlation with Sleep Apnea?
**Answer:** In clinical pathology, Obstructive Sleep Apnea (OSA) is strongly associated with excessive soft tissue in the upper airway (elevated BMI) and nocturnal sympathetic nervous system activation that drives systemic hypertension (elevated systolic/diastolic blood pressure). The ML model autonomously identified this clinical relationship: Sleep Apnea was predicted overwhelmingly in patients with BMI > 25 and systolic BP > 130 mmHg.

#### Q30: What software engineering best practices were incorporated in this project?
**Answer:**
- **Modular code structure:** Dedicated packages (`src/preprocessing.py`, `train_model.py`, `app.py`).
- **Automated test suite:** Comprehensive `pytest` test suite with 100% pass rate covering schema validation, edge cases, malformed BP inputs, and missing model behavior.
- **Reproducibility:** Fixed random seeds (`random_state=42`), frozen dependencies in `requirements.txt`, and automated plot generation scripts.
- **Version control:** Clean `.gitignore` excluding caches/virtualenvs and ready for GitHub.
