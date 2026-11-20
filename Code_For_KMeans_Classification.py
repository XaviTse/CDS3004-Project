import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

project_dir = Path(r"C:\Users\Isaac\OneDrive\Desktop\CDS3004_Project")
csv_path = project_dir / "hour.csv" if (project_dir / "hour.csv").exists() else Path("hour.csv")
hour_df = pd.read_csv(csv_path)

morning = hour_df[(hour_df["hr"].isin([7, 8])) & (hour_df["workingday"] == 1)].copy()

q75 = morning["cnt"].quantile(0.75)
morning["surge"] = (morning["cnt"] > q75).astype(int)


hr_stats = morning.groupby("hr")[["registered", "casual", "cnt"]].mean()
hours = hr_stats.index.tolist()
hour_labels = [f"{h:02d}:00" for h in hours]

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(
    hour_labels,
    hr_stats["registered"],
    label="Registered (Commuters)",
    color="#1f77b4",
    width=0.5,
)
ax.bar(
    hour_labels,
    hr_stats["casual"],
    bottom=hr_stats["registered"],
    label="Casual Riders",
    color="#ff7f0e",
    width=0.5,
)
ax.axhline(
    q75,
    color="#d62728",
    linestyle="--",
    linewidth=1.8,
    label=f"75th Percentile Surge Threshold ({q75:.2f})",
)

for idx, total in enumerate(hr_stats["cnt"]):
    if abs(total - q75) < 35:
        ax.text(
            idx,
            total - 22,
            f"{total:.1f}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
    else:
        ax.text(
            idx,
            total + 10,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

ax.set_title(
    "Working-Day Morning Commute Mean Ridership (07:00 - 09:00)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("Mean Hourly Rentals (cnt)", fontsize=11)
ax.set_ylim(0, max(hr_stats["cnt"].max(), q75) * 1.28)
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("graph1_morning_hourly_ridership.png", dpi=300, bbox_inches="tight")
plt.show()


baseline = morning["surge"].mean() * 100

segments = {
    "Rain / Snow / Mist (weathersit = 3)": (
        morning[morning["weathersit"] == 3]["surge"].mean() * 100
    ),
    "Year 2011 Working Morning (yr = 0)": (
        morning[morning["yr"] == 0]["surge"].mean() * 100
    ),
    "7:00 AM Working Day (hr = 7)": (
        morning[morning["hr"] == 7]["surge"].mean() * 100
    ),
    "Overall Working Morning Baseline (07:00-09:00)": baseline,
    "Clear / Few Clouds (weathersit = 1)": (
        morning[morning["weathersit"] == 1]["surge"].mean() * 100
    ),
    "Year 2012 Working Morning (yr = 1)": (
        morning[morning["yr"] == 1]["surge"].mean() * 100
    ),
    "8:00 AM Working Day (hr = 8)": (
        morning[morning["hr"] == 8]["surge"].mean() * 100
    ),
}

seg_labels = list(segments.keys())
seg_values = list(segments.values())
bar_colors = [
    "#7f7f7f" if "Baseline" in k else ("#2ca02c" if v >= baseline else "#d62728")
    for k, v in segments.items()
]

fig, ax = plt.subplots(figsize=(9.2, 5.0))
bars = ax.barh(seg_labels, seg_values, color=bar_colors)

for bar, val in zip(bars, seg_values):
    ax.text(
        val + 1.2,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center",
        ha="left",
        fontsize=10,
        fontweight="bold",
    )

ax.set_title(
    f"Morning Commute Surge Rate (cnt > {q75:.2f}) Across Segments",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Share of Working-Day Morning Hours in Surge (%)", fontsize=11)
ax.set_xlim(0, max(seg_values) * 1.15)
ax.grid(axis="x", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("graph2_morning_surge_rates.png", dpi=300, bbox_inches="tight")
plt.show()


cluster_features = ["hr", "temp", "hum", "windspeed"]
scaler = StandardScaler()
X_clust = scaler.fit_transform(morning[cluster_features])

ks = list(range(2, 8))
sses = []
for k in ks:
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_temp.fit(X_clust)
    sses.append(km_temp.inertia_)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(
    ks,
    sses,
    marker="o",
    linewidth=2.2,
    markersize=7,
    color="#1f77b4",
    label="SSE (Inertia)",
)

k4_idx = ks.index(4)
ax.scatter(
    [4],
    [sses[k4_idx]],
    color="#d62728",
    s=90,
    zorder=5,
    label="Elbow Candidate (K = 4)",
)
ax.annotate(
    f"K=4 (SSE={sses[k4_idx]:.0f})",
    xy=(4, sses[k4_idx]),
    xytext=(4.3, sses[k4_idx] + 500),
    arrowprops=dict(arrowstyle="->", color="#d62728"),
    fontsize=10,
    fontweight="bold",
    color="#d62728",
)

ax.set_title(
    "K-Means SSE Elbow Curve (Morning Weather & Hour Features)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Number of Clusters (K)", fontsize=11)
ax.set_ylabel("Within-Cluster Sum of Squares (SSE)", fontsize=11)
ax.set_xticks(ks)
ax.legend(loc="upper right")
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig("graph3_kmeans_elbow.png", dpi=300, bbox_inches="tight")
plt.show()

# Fit K=4 and print cluster profiles
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
morning["cluster_k4"] = kmeans_4.fit_predict(X_clust)

clust_summary = (
    morning.groupby("cluster_k4")
    .agg(
        count=("cnt", "count"),
        mean_hr=("hr", "mean"),
        mean_temp=("temp", "mean"),
        mean_hum=("hum", "mean"),
        mean_wind=("windspeed", "mean"),
        mean_cnt=("cnt", "mean"),
        surge_rate_pct=("surge", lambda s: s.mean() * 100),
    )
    .sort_values("surge_rate_pct", ascending=False)
)
print("=== K-Means (K=4) Cluster Profiling ===")
print(clust_summary.round(2))


feature_cols = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
X = morning[feature_cols]
y = morning["surge"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

print("\n=== Classification Model Comparison (5-Fold CV + 80/20 Holdout Test) ===")
for name, clf in models.items():
    cv_scores = cross_validate(
        clf, X_train, y_train, cv=cv, scoring=["f1", "roc_auc", "accuracy"]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print(f"\nModel: {name}")
    print(
        f"  5-Fold CV F1     : {cv_scores['test_f1'].mean():.4f} +/- {cv_scores['test_f1'].std():.4f}"
    )
    print(
        f"  5-Fold CV ROC-AUC: {cv_scores['test_roc_auc'].mean():.4f} +/- {cv_scores['test_roc_auc'].std():.4f}"
    )
    print(f"  Test Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test Precision   : {precision_score(y_test, y_pred):.4f}")
    print(f"  Test Recall      : {recall_score(y_test, y_pred):.4f}")
    print(f"  Test F1-Score    : {f1_score(y_test, y_pred):.4f}")
    print(f"  Test ROC-AUC     : {roc_auc_score(y_test, y_prob):.4f}")
    print("  Confusion Matrix :\n", confusion_matrix(y_test, y_pred))

dt_model = models["Decision Tree (max_depth=4)"]
print("\n=== Decision Tree (max_depth=4) Rules ===")
print(export_text(dt_model, feature_names=feature_cols))

rf_model = models["Random Forest (100 trees)"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)
print("\n=== Random Forest Feature Importances ===")
print(importances.round(4))
