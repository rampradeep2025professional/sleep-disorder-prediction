"""Model training, evaluation, comparison, and persistence script.

Compares:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors Classifier
5. Gradient Boosting Classifier

Evaluates models using 5-fold Stratified CV on training data and test set metrics
(Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Per-Class scores).
Saves the best end-to-end Pipeline to model/sleep_disorder_pipeline.joblib.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.preprocessing import (
    load_and_clean_data,
    build_full_pipeline,
    TARGET_CLASSES,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
DATA_PATH = "data/sleep_health.csv"
MODEL_DIR = "model"
IMAGES_DIR = "images/generated_analysis_plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def train_and_evaluate():
    print("=" * 60)
    print("SLEEP DISORDER CLASSIFICATION: MODEL TRAINING & COMPARISON")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1] Loading dataset from '{DATA_PATH}'...")
    X, y = load_and_clean_data(DATA_PATH)
    print(f"Total dataset shape: {X.shape}")
    print(f"Class counts:\n{y.value_counts()}")

    # 2. Stratified Train / Test Split
    print(f"\n[2] Performing stratified train/test split (test_size={TEST_SIZE}, seed={RANDOM_STATE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # 3. Define candidate algorithms
    candidates = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=5,
            min_samples_split=4,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=6,
            min_samples_split=4,
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    trained_pipelines = {}
    reports = {}
    conf_matrices = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n[3] Evaluating candidate models...")
    print("-" * 60)

    for name, clf in candidates.items():
        print(f"--> Training and evaluating: {name}")
        pipeline = build_full_pipeline(clf)

        # Cross-validation on training set to prevent leakage
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_macro"],
            n_jobs=-1,
        )
        cv_acc_mean = cv_scores["test_accuracy"].mean()
        cv_f1_macro_mean = cv_scores["test_f1_macro"].mean()

        # Fit on entire training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Evaluate on held-out test set
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reports[name] = rep
        cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
        conf_matrices[name] = cm

        results.append({
            "Model": name,
            "CV Accuracy": round(cv_acc_mean, 4),
            "CV Macro F1": round(cv_f1_macro_mean, 4),
            "Test Accuracy": round(acc, 4),
            "Test Macro Precision": round(macro_prec, 4),
            "Test Macro Recall": round(macro_rec, 4),
            "Test Macro F1": round(macro_f1, 4),
            "Test Weighted F1": round(weighted_f1, 4),
        })

    # 4. Summary comparison table
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["Test Macro F1", "CV Macro F1"], ascending=False).reset_index(drop=True)
    print("\n[4] MODEL COMPARISON SUMMARY:")
    print(df_results.to_string(index=False))

    comparison_csv_path = os.path.join(MODEL_DIR, "model_comparison.csv")
    df_results.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved comparison table to '{comparison_csv_path}'")

    # 5. Select Best Model (highest Test Macro F1)
    best_model_name = df_results.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = df_results.iloc[0].to_dict()
    best_rep = reports[best_model_name]
    best_cm = conf_matrices[best_model_name]

    print(f"\n[5] BEST MODEL SELECTED: '{best_model_name}'")
    print(f"    Test Accuracy: {best_metrics['Test Accuracy']:.4f}")
    print(f"    Test Macro F1: {best_metrics['Test Macro F1']:.4f}")

    # 6. Save Best Pipeline
    best_pipeline_path = os.path.join(MODEL_DIR, "sleep_disorder_pipeline.joblib")
    joblib.dump(best_pipeline, best_pipeline_path)
    print(f"\n[6] Saved best pipeline to '{best_pipeline_path}'")

    # 7. Save detailed metrics JSON
    metrics_metadata = {
        "best_model": best_model_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "classes": list(best_pipeline.classes_),
        "summary_metrics": best_metrics,
        "classification_report": best_rep,
        "confusion_matrix": best_cm.tolist(),
        "all_models_comparison": df_results.to_dict(orient="records"),
    }

    metrics_json_path = os.path.join(MODEL_DIR, "model_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_metadata, f, indent=2)
    print(f"Saved evaluation metrics & metadata to '{metrics_json_path}'")

    # 8. Visualizations: Model Comparison & Confusion Matrix
    print("\n[7] Generating performance plots...")
    # 8a. Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    plot_df = pd.melt(
        df_results,
        id_vars=["Model"],
        value_vars=["Test Accuracy", "Test Macro F1", "CV Macro F1"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Model Comparison: Accuracy & Macro F1 Scores")
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    comparison_plot_path = os.path.join(IMAGES_DIR, "model_comparison.png")
    plt.savefig(comparison_plot_path, dpi=300)
    plt.close()

    # 8b. Confusion Matrix of Best Model
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        best_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=best_pipeline.classes_,
        yticklabels=best_pipeline.classes_,
    )
    plt.title(f"Confusion Matrix: {best_model_name} (Test Set)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual True Class")
    plt.tight_layout()
    cm_plot_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to '{cm_plot_path}'")
    print(f"Saved model comparison plot to '{comparison_plot_path}'")

    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION COMPLETE")
    print("=" * 60)
    return best_model_name, best_metrics


if __name__ == "__main__":
    train_and_evaluate()
