
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
from config import (FIG_DIR, TABLE_DIR, IMU_CHANNELS,
                    BEHAVIOR_ORDER, ROAD_ORDER, DRIVER_META,)

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def output_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150,
                     "savefig.dpi": 300,
                     "font.size": 10,
                     "axes.titlesize": 11,
                     "figure.facecolor": "white",})

LABEL_MAP = {b: i for i, b in enumerate(BEHAVIOR_ORDER)}
NORMAL_IDX = LABEL_MAP["NORMAL"]

# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  participant and vehicle metadata  (thesis Table 1)
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

def participant_metadata(trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for driver in sorted(trips["driver"].unique()):
        d = trips[trips["driver"] == driver]
        row = {"Driver": driver,
               "Age": DRIVER_META.get(driver, {}).get("age", ""),
               "Vehicle": DRIVER_META.get(driver, {}).get("vehicle", ""),}
        for road in ROAD_ORDER:
            for beh in BEHAVIOR_ORDER:
                n = len(d[(d["behavior"] == beh) & (d["road"] == road)])
                row[f"{road[:3]}_{beh[:3]}"] = n if n > 0 else "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    out = TABLE_DIR / "metadata.csv"
    df.to_csv(out, index=False)
    return df


# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  trip duration  (thesis Table 2 + Figure 2)
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

def trip_durations(trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for beh in BEHAVIOR_ORDER:
        for road in ROAD_ORDER:
            sub = trips[(trips["behavior"] == beh) & (trips["road"] == road)]["duration_min"].dropna()
            rows.append({
                "Behavior": beh,
                "Road": road,
                "N_trips": len(sub),
                "Mean_min": round(sub.mean(), 1) if len(sub) else np.nan,
                "SD_min": round(sub.std(), 1) if len(sub) else np.nan,
                "Min_min": round(sub.min(), 1) if len(sub) else np.nan,
                "Max_min": round(sub.max(), 1) if len(sub) else np.nan,
                "Total_min": round(sub.sum(), 1) if len(sub) else np.nan,})
    df = pd.DataFrame(rows)
    out = TABLE_DIR / "durations.csv"
    df.to_csv(out, index=False)
    return df


def duration_boxplots(trips: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, road in enumerate(ROAD_ORDER):
        sub = trips[trips["road"] == road]
        sns.boxplot(
            data=sub, x="behavior", y="duration_min", order=BEHAVIOR_ORDER,
            palette="Set2", ax=axes[i], width=0.7,)
        axes[i].set_title(road.title(), fontsize=16)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Duration (min)" if i == 0 else "", fontsize=16)
        axes[i].tick_params(axis='x', labelsize=16)
        axes[i].tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    out = FIG_DIR / "duration_boxplots.png"
    fig.savefig(out)
    plt.close(fig)

# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  representative 30s time series  (thesis Figure 3)
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

def timeseries(all_accel: pd.DataFrame) -> None:
    output_dirs()
    channels = [c for c in IMU_CHANNELS if c in all_accel.columns]
    amp_scale = 0.9
    offset_step = 1.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    plt.subplots_adjust(top=0.70, bottom=0.18, wspace=0.08)
    legend_handles = []
    legend_labels = []

    for i, beh in enumerate(BEHAVIOR_ORDER):
        ax = axes[i]
        sub = all_accel[
            (all_accel["behavior"] == beh) &
            (all_accel["road"] == "MOTORWAY")]
        if sub.empty:
            ax.set_title(f"{beh.title()} (no data)", fontsize=20)
            continue

        trip_id = sub["trip_id"].iloc[0]
        trip = sub[sub["trip_id"] == trip_id].sort_values("timestamp")
        t0 = trip["timestamp"].iloc[len(trip) // 3]
        window = trip[(trip["timestamp"] >= t0) & (trip["timestamp"] < t0 + 30)]
        if window.empty:
            window = trip.head(300)

        t = window["timestamp"] - window["timestamp"].iloc[0]

        for j, ch in enumerate(channels):
            y = window[ch].to_numpy(dtype=float)
            mu = np.nanmean(y)
            sd = np.nanstd(y)
            y_scaled = (
                ((y - mu) / sd) * amp_scale
                if not (np.isclose(sd, 0) or np.isnan(sd))
                else np.zeros_like(y))

            offset = (len(channels) - 1 - j) * offset_step
            line, = ax.plot(t, y_scaled + offset, linewidth=1.4, label=ch)
            if i == 0:
                legend_handles.append(line)
                legend_labels.append(ch)

        ax.set_title("")
        ax.set_xlabel(beh.title(), fontsize=20, labelpad=10)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_yticks([])
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_ylabel("Standardized sensor value", fontsize=20)

        seg_ax = ax.inset_axes([0.0, 1.02, 1.0, 0.18])
        seg_ax.set_xlim(0, 30)
        seg_ax.set_ylim(0, 1)
        seg_ax.set_xticks([])
        seg_ax.set_yticks([])
        for sp in seg_ax.spines.values():
            sp.set_visible(False)

        row_y = {0: 0.6, 1: 0.4}
        bar_h = 0.075
        colors = {0: "#1b1b1b", 1: "#adadad"}
        starts = np.arange(0, 25.01, 2.5)

        for k, s in enumerate(starts):
            row = k % 2
            rect = mpatches.Rectangle(
                (s, row_y[row] - bar_h), 5.0, 2 * bar_h,
                linewidth=2.0, edgecolor="white",
                facecolor=colors[row], alpha=0.80, zorder=3,)
            seg_ax.add_patch(rect)

        seg_ax.set_title(
            "Segmentation · 5s windows · 50% overlap" if i == 1 else "",
            fontsize=20, color="#555555", style="italic", pad=6)

    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", ncol=len(channels), fontsize=20,
        frameon=True, bbox_to_anchor=(0.5, 1.01),
        borderpad=0.7, columnspacing=1.2, handlelength=1.8,)

    out = FIG_DIR / "timeseries.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  cross-channel correlation matrices  (thesis Figure 4)
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

def correlation_matrices(all_accel: pd.DataFrame):
    channels = [c for c in IMU_CHANNELS if c in all_accel.columns]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7),
                             gridspec_kw={"width_ratios": [1, 1, 1.18]})

    for i, beh in enumerate(BEHAVIOR_ORDER):
        sub = all_accel[all_accel["behavior"] == beh][channels].dropna()
        corr = sub.corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, ax=axes[i],
            cbar=i == 2, cbar_kws={"shrink": 0.9},
            annot_kws={"size": 26},)
        if i == 2:
            cbar = axes[i].collections[0].colorbar
            cbar.ax.tick_params(labelsize=28)
        axes[i].set_title(beh.title(), fontsize=26, fontweight="bold", pad=8)
        axes[i].tick_params(labelsize=22)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        if i > 0:
            axes[i].set_yticklabels([])

    plt.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.02, wspace=0.05)
    out = FIG_DIR / "correlation_matrices.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  sampling rate summary  (thesis Table 3)
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

def table_sampling_rates(rate_df: pd.DataFrame) -> pd.DataFrame:
    row = pd.DataFrame([{
        "N trips": len(rate_df),
        "Mean Hz": round(rate_df["mean_hz"].mean(), 2),
        "SD Hz": round(rate_df["mean_hz"].std(), 3),
        "Min Hz": round(rate_df["mean_hz"].min(), 2),
        "Max Hz": round(rate_df["mean_hz"].max(), 2),
        "Mean samples per trip": round(rate_df["n_samples"].mean(), 0),}])
    out = TABLE_DIR / "table_sampling_rates.csv"
    row.to_csv(out, index=False)
    return row


# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  modeling evaluation plots & tables (thesis Figure 5, Figure 6, Table 6, Table 10 + ECE)
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

def plot_calibration_curves(cv_results: dict, output_dir: Path = FIG_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"LogisticRegression": "#4a7fc1", "RandomForest": "#e07b39", "XGBoost": "#4CAF50"}
    fig, axes = plt.subplots(1, len(cv_results), figsize=(6 * len(cv_results), 4.5), sharey=True)
    if len(cv_results) == 1:
        axes = [axes]

    for i, (ax, (name, res)) in enumerate(zip(axes, cv_results.items())):
        y_true_bin = (res["all_true"] == NORMAL_IDX).astype(int)
        y_prob = res["all_proba"][:, NORMAL_IDX]
        frac_pos, mean_pred = calibration_curve(y_true_bin, y_prob, n_bins=10, strategy="uniform")
        ece = expected_calibration_error(y_true_bin, y_prob)

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", color=colors.get(name, "#555"),
                lw=2, ms=6, label=f"ECE = {ece:.3f}")
        ax.set_title(name, fontsize=18, fontweight="bold")
        ax.legend(fontsize=18, loc="upper left")
        ax.tick_params(axis="both", labelsize=16)

        if i == 0:
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        elif i == len(cv_results) - 1:
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            ax.set_xticks([0.2, 0.4, 0.6, 0.8])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.12, wspace=0.05)
    out = output_dir / "fig_calibration_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def score_distributions(person_scores: pd.DataFrame) -> None:

    COND_COLORS = {"NORMAL": "#4CAF50", "DROWSY": "#2196F3", "AGGRESSIVE": "#F44336",}
    models = ["LogisticRegression", "RandomForest", "XGBoost"]
    model_labels = {"LogisticRegression": "Logistic Regression",
                    "RandomForest": "Random Forest",
                    "XGBoost": "XGBoost",}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.patch.set_facecolor("white")

    for i, model_name in enumerate(models):
        ax = axes[i]
        ax.set_facecolor("white")
        ps = person_scores[person_scores["model"] == model_name]

        data = [ps[ps["behavior"] == beh]["mean_p_normal"].values for beh in BEHAVIOR_ORDER]
        positions = [1, 2, 3]
        colors = [COND_COLORS[b] for b in BEHAVIOR_ORDER]

        bp = ax.boxplot(
            data, positions=positions, patch_artist=True, widths=0.7,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 5, "linestyle": "none"},)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for flier, color in zip(bp["fliers"], colors):
            flier.set_markerfacecolor(color)
            flier.set_markeredgecolor(color)

        ax.set_title(model_labels[model_name], fontsize=20, fontweight="bold", pad=8)
        ax.set_xticks(positions)
        ax.set_xticklabels(["Normal", "Drowsy", "Aggressive"], fontsize=20)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.35, 3.65)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        if i == 0:
            ax.set_ylabel("Mean P(Normal)", fontsize=22)
            ax.tick_params(axis="y", labelsize=20)

    plt.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.08, wspace=0.08)
    out = FIG_DIR / "score_distributions.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lodo_performance(cv_results: dict, output_dir: Path = FIG_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    drivers = sorted(cv_results[list(cv_results.keys())[0]]["fold_results"]["held_out_driver"].unique())
    x = np.arange(len(drivers))
    width = 0.25
    colors = {"LogisticRegression": "#4a7fc1", "RandomForest": "#e07b39", "XGBoost": "#4CAF50"}
    fig, ax = plt.subplots(figsize=(7, 2.5))

    for k, (name, res) in enumerate(cv_results.items()):
        fold_df = res["fold_results"].set_index("held_out_driver")
        f1s = [fold_df.loc[d, "macro_f1"] for d in drivers]
        ax.bar(x + k * width, f1s, width, label=name, color=colors.get(name, "#888"), alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(drivers, fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.legend(fontsize=10, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.set_ylim(0, 1.05)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = output_dir / "lodo_performance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def build_summary_table(cv_results: dict) -> pd.DataFrame:
    rows = []
    for name, res in cv_results.items():
        fd = res["fold_results"]
        rows.append({"Model": name,
                     "Accuracy": f"{fd['accuracy'].mean():.3f} ({fd['accuracy'].std():.3f})",
                     "Macro F1": f"{fd['macro_f1'].mean():.3f} ({fd['macro_f1'].std():.3f})",
                     "F1 Normal": f"{fd['f1_normal'].mean():.3f} ({fd['f1_normal'].std():.3f})",
                     "F1 Drowsy": f"{fd['f1_drowsy'].mean():.3f} ({fd['f1_drowsy'].std():.3f})",
                     "F1 Aggressive": f"{fd['f1_aggressive'].mean():.3f} ({fd['f1_aggressive'].std():.3f})",
                     "ECE Normal": f"{fd['ece_normal'].mean():.3f} ({fd['ece_normal'].std():.3f})",})
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_true_bin[mask].mean() - y_prob[mask].mean())
    return float(ece)
