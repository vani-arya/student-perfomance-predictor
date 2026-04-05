"""
eda.py  (run as a script or convert to Jupyter notebook)
─────────────────────────────────────────────────────────
Exploratory Data Analysis for the Students Performance dataset.

Run: python notebook/eda.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Ensure src/ is on path ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────
print("Loading dataset...")
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "students.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    print("⚠️  data/students.csv not found — generating synthetic data")
    np.random.seed(42); n = 1000
    reading = np.clip(np.random.normal(69, 15, n).astype(int), 0, 100)
    writing = np.clip(np.random.normal(68, 15, n).astype(int), 0, 100)
    math    = np.clip(((reading + writing) / 2 + np.random.normal(0, 8, n)).astype(int), 0, 100)
    df = pd.DataFrame({
        "gender":                      np.random.choice(["male","female"], n),
        "race/ethnicity":              np.random.choice(["group A","group B","group C","group D","group E"], n),
        "parental level of education": np.random.choice(["high school","some college","associate's degree","bachelor's degree","master's degree"], n),
        "lunch":                       np.random.choice(["standard","free/reduced"], n),
        "test preparation course":     np.random.choice(["none","completed"], n),
        "reading score": reading, "writing score": writing, "math score": math,
    })

print(f"Shape: {df.shape}")
print(df.head())
print("\nBasic Stats:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ── 2. Score Distributions ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Score Distributions", fontsize=14, fontweight="bold")
for ax, col, color in zip(axes,
    ["math score", "reading score", "writing score"],
    ["#4C72B0", "#DD8452", "#55A868"]):
    ax.hist(df[col], bins=25, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.1f}")
    ax.set_title(col.title()); ax.set_xlabel("Score"); ax.set_ylabel("Count"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "score_distributions.png"), dpi=150)
plt.show(); print("✅ Score distributions saved")

# ── 3. Correlation Heatmap ───────────────────────────────────
plt.figure(figsize=(6, 4))
corr = df[["math score", "reading score", "writing score"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", mask=mask,
            linewidths=0.5, square=True)
plt.title("Correlation Between Scores", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "correlation_heatmap.png"), dpi=150)
plt.show(); print("✅ Correlation heatmap saved")

# ── 4. Scores by Gender ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Scores by Gender", fontweight="bold")
for ax, col in zip(axes, ["math score", "reading score", "writing score"]):
    sns.boxplot(data=df, x="gender", y=col, ax=ax, palette="pastel")
    ax.set_title(col.title())
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "scores_by_gender.png"), dpi=150)
plt.show(); print("✅ Gender boxplots saved")

# ── 5. Test Prep Impact ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Impact of Test Preparation Course", fontweight="bold")
for ax, col in zip(axes, ["math score", "reading score", "writing score"]):
    sns.violinplot(data=df, x="test preparation course", y=col, ax=ax,
                   palette=["#E07B7B", "#7BB8E0"], inner="box")
    ax.set_title(col.title())
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "test_prep_impact.png"), dpi=150)
plt.show(); print("✅ Test prep violin plots saved")

# ── 6. Parental Education vs Math Score ──────────────────────
plt.figure(figsize=(10, 5))
order = ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
sns.barplot(data=df, x="parental level of education", y="math score",
            order=order, palette="coolwarm", errorbar="sd")
plt.title("Average Math Score by Parental Education Level", fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "parental_edu_impact.png"), dpi=150)
plt.show(); print("✅ Parental education impact saved")

# ── 7. Pairplot ──────────────────────────────────────────────
pair_df = df[["math score", "reading score", "writing score", "gender"]]
g = sns.pairplot(pair_df, hue="gender", plot_kws={"alpha": 0.5}, diag_kind="kde")
g.fig.suptitle("Pairplot of Scores by Gender", y=1.02, fontweight="bold")
plt.savefig(os.path.join(SAVE_DIR, "pairplot.png"), dpi=120)
plt.show(); print("✅ Pairplot saved")

print("\n🎉 EDA Complete! All plots saved to artifacts/")
