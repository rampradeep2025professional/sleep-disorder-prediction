"""Data preprocessing and feature engineering module for Sleep Disorder Classification.

This module provides reusable transformers, data cleaning functions,
and preprocessing pipelines using Scikit-Learn standards to avoid data leakage
and ensure training-inference consistency.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Canonical target classes
TARGET_CLASSES: List[str] = ["None", "Insomnia", "Sleep Apnea"]

# Raw expected feature columns for prediction (excluding Person ID and Sleep Disorder)
EXPECTED_RAW_FEATURES: List[str] = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps",
]

# Engineered feature names after FeatureEngineer transformation
NUMERICAL_FEATURES: List[str] = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "Systolic_BP",
    "Diastolic_BP",
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Occupation",
    "BMI Category",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-learn Transformer for Sleep Health domain feature engineering.

    Tasks performed:
    1. Removes identifier columns such as 'Person ID'.
    2. Standardizes 'BMI Category' ('Normal Weight' -> 'Normal').
    3. Parses compound 'Blood Pressure' string (e.g. '120/80') into numerical
       'Systolic_BP' and 'Diastolic_BP'.
    4. Handles fallback if Systolic_BP and Diastolic_BP are already numeric columns.
    """

    def __init__(self, default_systolic: float = 120.0, default_diastolic: float = 80.0):
        self.default_systolic = default_systolic
        self.default_diastolic = default_diastolic

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()

        # Remove identifier if present
        if "Person ID" in X.columns:
            X = X.drop(columns=["Person ID"])

        # Standardize BMI Category
        if "BMI Category" in X.columns:
            X["BMI Category"] = (
                X["BMI Category"]
                .astype(str)
                .str.strip()
                .replace({"Normal Weight": "Normal"})
            )

        # Parse Blood Pressure if present as string
        if "Blood Pressure" in X.columns:
            bp_series = X["Blood Pressure"].astype(str)
            splits = bp_series.str.split("/", expand=True)

            if splits.shape[1] >= 2:
                systolic = pd.to_numeric(splits[0], errors="coerce").fillna(self.default_systolic)
                diastolic = pd.to_numeric(splits[1], errors="coerce").fillna(self.default_diastolic)
            else:
                systolic = pd.Series(self.default_systolic, index=X.index)
                diastolic = pd.Series(self.default_diastolic, index=X.index)

            X["Systolic_BP"] = systolic.astype(float)
            X["Diastolic_BP"] = diastolic.astype(float)
            X = X.drop(columns=["Blood Pressure"])
        else:
            # If Blood Pressure column is absent, check if Systolic_BP and Diastolic_BP exist
            if "Systolic_BP" not in X.columns:
                X["Systolic_BP"] = float(self.default_systolic)
            if "Diastolic_BP" not in X.columns:
                X["Diastolic_BP"] = float(self.default_diastolic)

        # Ensure all expected numerical columns exist and are numeric
        for col in NUMERICAL_FEATURES:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                X[col] = np.nan

        # Ensure categorical columns exist and are string
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype(str)
            else:
                X[col] = "Missing"

        # Return only the engineered columns in defined order
        all_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        return X[all_cols]


def build_preprocessor() -> ColumnTransformer:
    """Creates a ColumnTransformer for numerical scaling and categorical encoding.

    Numerical pipeline: SimpleImputer(median) -> StandardScaler
    Categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(ignore unknown)
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_full_pipeline(classifier) -> Pipeline:
    """Builds a complete end-to-end Pipeline from raw DataFrame to prediction."""
    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("engineer", FeatureEngineer()),
        ("preprocess", preprocessor),
        ("classifier", classifier),
    ])
    return pipeline


def load_and_clean_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads sleep health dataset and formats features and target.

    In the official dataset:
    - Missing target values (NaN) denote 'None' (no sleep disorder).
    - Identifiers ('Person ID') are retained in X for the pipeline to drop safely,
      or dropped during splitting.
    """
    df = pd.read_csv(filepath)

    if "Sleep Disorder" not in df.columns:
        raise ValueError(f"'Sleep Disorder' column not found in dataset at {filepath}")

    # Standardize target
    y = df["Sleep Disorder"].fillna("None").astype(str).str.strip()

    # Features
    X = df.drop(columns=["Sleep Disorder"])

    return X, y
