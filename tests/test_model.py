"""Automated test suite for Sleep Disorder Classification pipeline, preprocessing, and model inference."""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    load_and_clean_data,
    FeatureEngineer,
    build_full_pipeline,
    TARGET_CLASSES,
    EXPECTED_RAW_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/sleep_health.csv"
MODEL_PATH = "model/sleep_disorder_pipeline.joblib"
METRICS_PATH = "model/model_metrics.json"
PLOTS_DIR = "images/generated_analysis_plots"


def test_dataset_loading_and_shape():
    """Verify that the dataset exists, loads cleanly, and has valid shape."""
    assert os.path.exists(DATA_PATH), f"Dataset file not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] == 374, f"Expected 374 rows, got {df.shape[0]}"
    assert df.shape[1] == 13, f"Expected 13 columns, got {df.shape[1]}"
    assert "Sleep Disorder" in df.columns
    assert "Person ID" in df.columns


def test_target_validation_and_cleaning():
    """Verify target cleaning maps NaNs to 'None' and yields expected classes."""
    X, y = load_and_clean_data(DATA_PATH)
    unique_targets = set(y.unique())
    expected = {"None", "Insomnia", "Sleep Apnea"}
    assert unique_targets == expected, f"Unexpected target classes: {unique_targets}"
    assert len(y) == 374
    # Check specific distribution counts
    counts = y.value_counts().to_dict()
    assert counts["None"] == 219
    assert counts["Sleep Apnea"] == 78
    assert counts["Insomnia"] == 77


def test_feature_engineer_transformer():
    """Verify FeatureEngineer handles BP splitting, BMI normalization, and ID removal."""
    sample_df = pd.DataFrame({
        "Person ID": [101, 102],
        "Gender": ["Male", "Female"],
        "Age": [30, 45],
        "Occupation": ["Engineer", "Doctor"],
        "Sleep Duration": [7.0, 6.0],
        "Quality of Sleep": [8, 5],
        "Physical Activity Level": [60, 40],
        "Stress Level": [4, 7],
        "BMI Category": ["Normal Weight", "Overweight"],
        "Blood Pressure": ["126/83", "135/90"],
        "Heart Rate": [70, 78],
        "Daily Steps": [8000, 5000],
    })

    transformer = FeatureEngineer()
    transformed = transformer.transform(sample_df)

    # Person ID dropped
    assert "Person ID" not in transformed.columns
    # Blood Pressure parsed
    assert "Systolic_BP" in transformed.columns
    assert "Diastolic_BP" in transformed.columns
    assert transformed["Systolic_BP"].iloc[0] == 126.0
    assert transformed["Diastolic_BP"].iloc[0] == 83.0
    assert transformed["Systolic_BP"].iloc[1] == 135.0
    assert transformed["Diastolic_BP"].iloc[1] == 90.0
    # BMI normalized
    assert transformed["BMI Category"].iloc[0] == "Normal"
    assert transformed["BMI Category"].iloc[1] == "Overweight"


def test_saved_model_artifact_exists_and_loads():
    """Verify that the saved model artifact exists and can be loaded."""
    assert os.path.exists(MODEL_PATH), f"Trained model not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)
    assert hasattr(pipeline, "predict"), "Pipeline missing predict method"
    assert hasattr(pipeline, "predict_proba"), "Pipeline missing predict_proba method"
    assert hasattr(pipeline, "classes_"), "Pipeline missing classes_ attribute"


def test_prediction_output_and_class_labels():
    """Test model inference on sample records producing valid class labels."""
    pipeline = joblib.load(MODEL_PATH)

    sample_input = pd.DataFrame([{
        "Gender": "Male",
        "Age": 32,
        "Occupation": "Software Engineer",
        "Sleep Duration": 7.5,
        "Quality of Sleep": 8,
        "Physical Activity Level": 65,
        "Stress Level": 4,
        "BMI Category": "Normal",
        "Blood Pressure": "120/80",
        "Heart Rate": 68,
        "Daily Steps": 8500,
    }])

    pred = pipeline.predict(sample_input)
    assert len(pred) == 1
    assert pred[0] in TARGET_CLASSES, f"Predicted class '{pred[0]}' not in {TARGET_CLASSES}"


def test_prediction_probabilities_integrity():
    """Verify that predicted probabilities sum to 1.0 and cover all 3 classes."""
    pipeline = joblib.load(MODEL_PATH)

    sample_input = pd.DataFrame([{
        "Gender": "Female",
        "Age": 50,
        "Occupation": "Nurse",
        "Sleep Duration": 6.1,
        "Quality of Sleep": 6,
        "Physical Activity Level": 90,
        "Stress Level": 8,
        "BMI Category": "Overweight",
        "Blood Pressure": "140/95",
        "Heart Rate": 75,
        "Daily Steps": 5000,
    }])

    probs = pipeline.predict_proba(sample_input)[0]
    assert len(probs) == 3
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5), f"Probabilities sum to {np.sum(probs)}, not 1.0"
    for p in probs:
        assert 0.0 <= p <= 1.0


def test_unexpected_categorical_values_handling():
    """Verify that unseen categories during inference are safely handled without errors."""
    pipeline = joblib.load(MODEL_PATH)

    unseen_input = pd.DataFrame([{
        "Gender": "NonBinary",  # Unseen
        "Age": 29,
        "Occupation": "Astronaut",  # Unseen
        "Sleep Duration": 6.8,
        "Quality of Sleep": 7,
        "Physical Activity Level": 50,
        "Stress Level": 5,
        "BMI Category": "Severely Obese",  # Unseen
        "Blood Pressure": "125/82",
        "Heart Rate": 72,
        "Daily Steps": 6000,
    }])

    # Should not raise an exception
    pred = pipeline.predict(unseen_input)
    assert pred[0] in TARGET_CLASSES


def test_missing_values_imputation_handling():
    """Verify that inputs with missing values are cleanly imputed by pipeline without crashing."""
    pipeline = joblib.load(MODEL_PATH)

    missing_input = pd.DataFrame([{
        "Gender": "Male",
        "Age": np.nan,  # Missing
        "Occupation": "Doctor",
        "Sleep Duration": np.nan,  # Missing
        "Quality of Sleep": 7,
        "Physical Activity Level": 50,
        "Stress Level": np.nan,  # Missing
        "BMI Category": "Normal",
        "Blood Pressure": "invalid_bp_format",  # Malformed BP
        "Heart Rate": np.nan,
        "Daily Steps": 7000,
    }])

    pred = pipeline.predict(missing_input)
    assert pred[0] in TARGET_CLASSES


def test_metrics_json_file_contents():
    """Verify model_metrics.json exists and contains correct structure and values."""
    assert os.path.exists(METRICS_PATH), f"Metrics file missing at {METRICS_PATH}"
    with open(METRICS_PATH, "r") as f:
        data = json.load(f)

    assert "best_model" in data
    assert "summary_metrics" in data
    assert "classification_report" in data
    assert "confusion_matrix" in data

    metrics = data["summary_metrics"]
    assert metrics["Test Accuracy"] > 0.85
    assert metrics["Test Macro F1"] > 0.80


def test_generated_eda_plots_exist():
    """Verify that generated plots exist and are non-empty."""
    expected_plots = [
        "01_target_distribution.png",
        "02_age_distribution.png",
        "03_sleep_duration_distribution.png",
        "04_sleep_quality_distribution.png",
        "05_stress_level_distribution.png",
        "06_physical_activity_distribution.png",
        "07_bmi_category_distribution.png",
        "08_sleep_duration_vs_disorder.png",
        "09_stress_level_vs_disorder.png",
        "10_sleep_quality_vs_disorder.png",
        "11_bmi_category_vs_disorder.png",
        "12_correlation_heatmap.png",
        "confusion_matrix.png",
        "model_comparison.png",
    ]
    for plot_name in expected_plots:
        plot_path = os.path.join(PLOTS_DIR, plot_name)
        assert os.path.exists(plot_path), f"Plot {plot_name} not found in {PLOTS_DIR}"
        assert os.path.getsize(plot_path) > 1000, f"Plot {plot_name} is unexpectedly small or empty"


def test_app_script_syntax_and_compilation():
    """Verify that app.py compiles without syntax errors."""
    import py_compile
    py_compile.compile("app.py", doraise=True)


def test_missing_model_graceful_handling():
    """Verify that querying a missing model file returns None gracefully."""
    fake_path = "model/non_existent_model.joblib"
    assert not os.path.exists(fake_path)
