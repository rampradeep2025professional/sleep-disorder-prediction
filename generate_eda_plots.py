"""Exploratory Data Analysis script for Sleep Disorder Classification.

Generates 12 publication-quality visualizations and saves them to images/generated_analysis_plots/.
Calculates and logs genuine statistics from the dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
})

OUTPUT_DIR = "images/generated_analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load and prepare dataset
df = pd.read_csv("data/sleep_health.csv")
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None").astype(str).str.strip()
df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

# Split Blood Pressure into Systolic and Diastolic
bp_split = df["Blood Pressure"].str.split("/", expand=True)
df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors="coerce")
df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors="coerce")

print("=== DATASET OVERVIEW ===")
print("Total records:", len(df))
print("Class counts:\n", df["Sleep Disorder"].value_counts())
print("\nBMI Category counts:\n", df["BMI Category"].value_counts())

palette_target = {"None": "#2ecc71", "Insomnia": "#e67e22", "Sleep Apnea": "#e74c3c"}

# 1. Target class distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x="Sleep Disorder", palette=palette_target, order=["None", "Insomnia", "Sleep Apnea"])
plt.title("1. Sleep Disorder Class Distribution")
plt.xlabel("Sleep Disorder Category")
plt.ylabel("Number of Individuals")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png", dpi=300)
plt.close()

# 2. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], kde=True, color="#3498db", bins=15)
plt.axvline(df["Age"].median(), color="red", linestyle="--", label=f"Median: {df['Age'].median():.0f} yrs")
plt.title("2. Age Distribution of Participants")
plt.xlabel("Age (Years)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_age_distribution.png", dpi=300)
plt.close()

# 3. Sleep duration distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Sleep Duration"], kde=True, color="#9b59b6", bins=12)
plt.axvline(df["Sleep Duration"].mean(), color="black", linestyle="--", label=f"Mean: {df['Sleep Duration'].mean():.2f} hrs")
plt.title("3. Sleep Duration Distribution")
plt.xlabel("Sleep Duration (Hours / Day)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_sleep_duration_distribution.png", dpi=300)
plt.close()

# 4. Sleep quality distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x="Quality of Sleep", color="#1abc9c")
plt.title("4. Quality of Sleep Distribution (Scale: 1-10)")
plt.xlabel("Subjective Quality of Sleep Score")
plt.ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=10, xytext=(0, 2), textcoords="offset points")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_sleep_quality_distribution.png", dpi=300)
plt.close()

# 5. Stress level distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x="Stress Level", color="#f39c12")
plt.title("5. Stress Level Distribution (Scale: 1-10)")
plt.xlabel("Stress Level Score")
plt.ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=10, xytext=(0, 2), textcoords="offset points")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_stress_level_distribution.png", dpi=300)
plt.close()

# 6. Physical activity distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Physical Activity Level"], kde=True, color="#27ae60", bins=12)
plt.axvline(df["Physical Activity Level"].mean(), color="black", linestyle="--", label=f"Mean: {df['Physical Activity Level'].mean():.1f} min/day")
plt.title("6. Physical Activity Level Distribution")
plt.xlabel("Physical Activity (Minutes / Day)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_physical_activity_distribution.png", dpi=300)
plt.close()

# 7. BMI category distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x="BMI Category", palette="Blues_d", order=["Normal", "Overweight", "Obese"])
plt.title("7. BMI Category Distribution")
plt.xlabel("Standardized BMI Category")
plt.ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_bmi_category_distribution.png", dpi=300)
plt.close()

# 8. Sleep duration versus disorder
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Sleep Disorder", y="Sleep Duration", palette=palette_target, order=["None", "Insomnia", "Sleep Apnea"])
plt.title("8. Sleep Duration by Sleep Disorder Status")
plt.xlabel("Sleep Disorder Category")
plt.ylabel("Sleep Duration (Hours)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_sleep_duration_vs_disorder.png", dpi=300)
plt.close()

# 9. Stress level versus disorder
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Sleep Disorder", y="Stress Level", palette=palette_target, order=["None", "Insomnia", "Sleep Apnea"])
plt.title("9. Stress Level by Sleep Disorder Status")
plt.xlabel("Sleep Disorder Category")
plt.ylabel("Stress Level Score (1-10)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_stress_level_vs_disorder.png", dpi=300)
plt.close()

# 10. Sleep quality versus disorder
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Sleep Disorder", y="Quality of Sleep", palette=palette_target, order=["None", "Insomnia", "Sleep Apnea"])
plt.title("10. Sleep Quality by Sleep Disorder Status")
plt.xlabel("Sleep Disorder Category")
plt.ylabel("Sleep Quality Score (1-10)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_sleep_quality_vs_disorder.png", dpi=300)
plt.close()

# 11. BMI category versus disorder
plt.figure(figsize=(9, 5.5))
ct = pd.crosstab(df["BMI Category"], df["Sleep Disorder"], normalize="index")[["None", "Insomnia", "Sleep Apnea"]] * 100
ax = ct.plot(kind="bar", stacked=True, color=[palette_target[c] for c in ["None", "Insomnia", "Sleep Apnea"]], figsize=(9, 5.5))
plt.title("11. Sleep Disorder Proportion Across BMI Categories")
plt.xlabel("BMI Category")
plt.ylabel("Percentage (%)")
plt.legend(title="Sleep Disorder", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_bmi_category_vs_disorder.png", dpi=300)
plt.close()

# 12. Correlation heatmap for appropriate numerical features
num_cols = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
            "Stress Level", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"]
corr_matrix = df[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("12. Correlation Heatmap of Numerical Attributes")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_correlation_heatmap.png", dpi=300)
plt.close()

print("\nSuccessfully generated all 12 EDA visualizations in:", OUTPUT_DIR)
