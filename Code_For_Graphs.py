import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pathlib import Path

project_dir = Path(__file__).resolve().parent
# Filter the morning hours
hour_df = pd.read_csv(project_dir / "hour.csv")
morning = hour_df[(hour_df["hr"].isin([7, 8])) & (hour_df["workingday"] == 1)].copy()
# Mark the surge hours
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
X_scaled = scaler.fit_transform(morning[cluster_features])

ks = list(range(2, 8))
sses = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sses.append(km.inertia_)

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
