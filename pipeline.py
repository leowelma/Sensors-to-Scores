# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  Imports
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from typing import Optional
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf

from config import (DATA_ROOT, ACCEL_COLS, GPS_COLS, DRIVER_META,
                    BEHAVIOR_ORDER, ROAD_ORDER, MANEUVER_TYPES,
                    IMU_CHANNELS, TABLE_DIR, FIG_DIR,)

from descriptives import (
    participant_metadata, trip_durations,
    duration_boxplots, timeseries,
    correlation_matrices, table_sampling_rates,
    plot_calibration_curves, plot_lodo_performance,
    build_summary_table, expected_calibration_error, NORMAL_IDX,
    score_distributions,)

RANDOM_STATE = 373885

# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  SECTION 1: DATA LOADING
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

# function to extract data from ROOT folder and return a DataFrame of trip metadata,
# functions to load accel, GPS, and semantic data for each trip
# Each row: driver, behavior, road, distance_km, trip_folder (Path), datetime
def discover_trips(data_root: Path = DATA_ROOT) -> pd.DataFrame:

    rows = []
    pat = re.compile(
        # folder naming format (Example: 20151111123124-25km-D1-NORMAL-MOTORWAY)
        r"^(\d{14})-([\d.]+)km?-(D\d)-(NORMAL\d?|DROWSY|AGGRESSIVE)-(MOTORWAY|SECONDARY)$",
        re.IGNORECASE,)

    # Loop through dataset folders
    for driver_dir in sorted(data_root.iterdir()):
        if not driver_dir.is_dir():
            continue
        for trip_dir in sorted(driver_dir.iterdir()):
            if not trip_dir.is_dir():
                continue
            m = pat.match(trip_dir.name)
            if m is None:
                warnings.warn(f"wrong folder format: {trip_dir.name}")
                continue

            dt_str, dist, driver, behavior, road = m.groups()
            behavior = re.sub(r"\d+$", "", behavior.upper())   # NORMAL1 → NORMAL
            rows.append({
                "driver": driver.upper(),
                "behavior": behavior.upper(),
                "road": road.upper(),
                "distance_km": float(dist),
                "datetime_str": dt_str,
                "trip_folder": trip_dir,})

    trips = pd.DataFrame(rows)

    # drver meta
    for col in ["age", "vehicle"]:
        trips[col] = trips["driver"].map(lambda d: DRIVER_META.get(d, {}).get(col))

    trips["behavior"] = pd.Categorical(trips["behavior"], categories=BEHAVIOR_ORDER, ordered=True)
    trips["road"] = pd.Categorical(trips["road"], categories=ROAD_ORDER, ordered=True)

    print(f"total trips: {len(trips)} trips across {trips['driver'].nunique()} drivers")
    return trips.reset_index(drop=True)

# helper function for accel and gyro data (RAW_ACCELEROMETERS.txt and RAW_GPS.txt)
def csv_helper(path: Path, col_names: list) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df.columns = col_names[:df.shape[1]]
    return df

# load accel data for one trip
def load_accel(trip_folder: Path) -> Optional[pd.DataFrame]:
    return csv_helper(trip_folder / "RAW_ACCELEROMETERS.txt", ACCEL_COLS)

# load SEMANTIC_FINAL.txt for one trip
# label names can be found in readme of the UAH-driveset reader file
def load_semantic_final(trip_folder: Path) -> dict:
    lines = (trip_folder / "SEMANTIC_FINAL.txt").read_text().splitlines()

    def _float(n):
        return float(lines[n - 1].strip().split()[0])

    return { "driving_time_min": _float(7), "score_total": _float(44), "accelerations": _float(45),
             "brakings": _float(46), "turnings": _float(47), "lane_weaving": _float(48),
             "lane_drifting": _float(49), "overspeeding": _float(50), "car_following": _float(51),
             "ratio_normal": _float(52), "ratio_drowsy": _float(53), "ratio_aggressive": _float(54),}

# load all accel data for all trips and puts it in one df
def load_all_accel(trips: pd.DataFrame, resample_hz: Optional[float] = 10.0) -> pd.DataFrame:
    frames = []
    for idx, row in trips.iterrows():
        df = load_accel(row["trip_folder"])
        df["driver"] = row["driver"]
        df["behavior"] = row["behavior"]
        df["road"] = row["road"]
        df["trip_id"] = idx

        t = df["timestamp"].values
        t_uniform = np.arange(t[0], t[-1], 1.0 / resample_hz)
        new_df = pd.DataFrame({"timestamp": t_uniform})
        for col in df.columns:
            if col not in ["timestamp", "driver", "behavior", "road", "trip_id"]:
                new_df[col] = np.interp(t_uniform, t, df[col].values.astype(float))
        new_df["driver"] = row["driver"]
        new_df["behavior"] = row["behavior"]
        new_df["road"] = row["road"]
        new_df["trip_id"] = idx
        df = new_df

        frames.append(df)

    all_accel = pd.concat(frames, ignore_index=True)
    print(f"total accelerometer samples: {len(all_accel):,}")
    return all_accel

# load all semantics for all trips and puts it in one df
def load_all_semantics(trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in trips.iterrows():
        sem = load_semantic_final(row["trip_folder"])
        sem["driver"] = row["driver"]
        sem["behavior"] = row["behavior"]
        sem["road"] = row["road"]
        sem["trip_id"] = idx
        rows.append(sem)

    df = pd.DataFrame(rows)
    print(f"loaded semantics for {len(df)}/{len(trips)} trips")
    return df

# computes trip durations (in min) from RAW accel timestamps
def compute_trip_durations(trips: pd.DataFrame) -> pd.DataFrame:
    durations = []
    for idx, row in trips.iterrows():
        df = load_accel(row["trip_folder"])
        t = df["timestamp"]
        durations.append((t.iloc[-1] - t.iloc[0]) / 60.0)
    trips = trips.copy()
    trips["duration_min"] = durations
    return trips

# computes sampling rates (in Hz) from RAW accel timestamps
def compute_sampling_rates(trips: pd.DataFrame) -> pd.DataFrame:
    rates = []
    for idx, row in trips.iterrows():
        df = load_accel(row["trip_folder"])
        t  = df["timestamp"].values
        duration = t[-1] - t[0]
        n_samples = len(df)
        hz = n_samples / duration if duration > 0 else np.nan
        rates.append({ "trip_id": idx,
                       "driver": row["driver"],
                       "behavior": row["behavior"],
                       "road": row["road"],
                       "mean_hz": round(hz, 3),
                       "n_samples": n_samples,
                       "duration_s": round(duration, 2),})
        return pd.DataFrame(rates)

# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ### SECTION 2: FEATURE ENGINEERING
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

# segmentation parameters
SAMPLE_HZ = 10
WINDOW_SEC = 5
OVERLAP = 0.5
WIN_SAMPLES = int(WINDOW_SEC * SAMPLE_HZ)
STRIDE = int(WIN_SAMPLES * (1 - OVERLAP))

#  ----  ----  ----  segmentation  ----  ----  ----

# function to segment by fixed window size and to return window data
def segment_trips(all_accel: pd.DataFrame) -> list:
    channels = [c for c in IMU_CHANNELS if c in all_accel.columns]
    windows  = []

    for trip_id, grp in all_accel.groupby("trip_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        arr = grp[channels].values.astype(float)
        driver = grp["driver"].iloc[0]
        behavior = str(grp["behavior"].iloc[0])
        road = str(grp["road"].iloc[0])
        n = len(arr)
        w_id = 0

        for start in range(0, n - WIN_SAMPLES + 1, STRIDE):
            chunk = arr[start : start + WIN_SAMPLES]
            windows.append({
                "data": chunk,
                "driver": driver,
                "behavior": behavior,
                "road": road,
                "trip_id": trip_id,
                "window_id": w_id,})
            w_id += 1

    print(f"segmentation resulted in {len(windows):} windows")
    return windows

#  ----  ----  ----  Feature group helpers  ----  ----  ----

# function to compute statistical moments for one channel in one window
def statistical_moments(x: np.ndarray, ch: str) -> dict:
    return { f"{ch}_mean": np.mean(x),
             f"{ch}_var": np.var(x, ddof=1),
             f"{ch}_sd": np.std(x, ddof=1),
             f"{ch}_skew": float(stats.skew(x)),
             f"{ch}_kurt": float(stats.kurtosis(x)),
             f"{ch}_rms": np.sqrt(np.mean(x ** 2)),
             f"{ch}_min": np.min(x),
             f"{ch}_max": np.max(x),
             f"{ch}_range": np.max(x) - np.min(x),
             f"{ch}_iqr": float(stats.iqr(x)),}

# autocorrelation at specified lags, zero-crossing rate, mean-crossing rate
def temporal_structure(x: np.ndarray, ch: str, lags: tuple = (1, 5, 10)) -> dict:

    feats = {}
    n = len(x)
    x_norm = x - np.mean(x)
    denom = np.sum(x_norm ** 2)

    for lag in lags:
        if denom == 0 or lag >= n:
            feats[f"{ch}_acf_lag{lag}"] = 0.0
        else:
            feats[f"{ch}_acf_lag{lag}"] = float(np.sum(x_norm[:-lag] * x_norm[lag:]) / denom)

    feats[f"{ch}_zcr"] = float(np.sum(np.diff(np.sign(x)) != 0) / (n - 1))
    mc = x - np.mean(x)
    feats[f"{ch}_mcr"] = float(np.sum(np.diff(np.sign(mc)) != 0) / (n - 1))
    return feats

# energy bands, dominant frequency, spectral entropy, centroid (FFT)
def frequency_domain(x: np.ndarray, ch: str, fs: float = SAMPLE_HZ) -> dict:
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2
    total_power = np.sum(power) + 1e-12
    feats = {}

    bands = {"low": (0.0, 1.0), "mid": (1.0, 3.0), "high": (3.0, fs / 2)}
    for band, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        feats[f"{ch}_energy_{band}"] = float(np.sum(power[mask]) / total_power)

    feats[f"{ch}_dom_freq"] = float(freqs[np.argmax(power)])

    p_norm = power / total_power
    p_norm = p_norm[p_norm > 0]
    feats[f"{ch}_spectral_entropy"]  = float(-np.sum(p_norm * np.log2(p_norm)))
    feats[f"{ch}_spectral_centroid"] = float(np.sum(freqs * power) / total_power)
    return feats

# sample entropy
def entropy_features(x: np.ndarray, ch: str, m: int = 2, r_scale: float = 0.2) -> dict:
    n = len(x)
    r = r_scale * np.std(x, ddof=1)
    if r == 0:
        return {f"{ch}_sample_entropy": np.nan}

    def _phi(m_):
        count = total = 0
        for i in range(n - m_):
            template = x[i : i + m_]
            for j in range(i + 1, n - m_):
                if np.max(np.abs(template - x[j : j + m_])) < r:
                    count += 1
            total += n - m_ - 1
        return count / max(total, 1)

    B = _phi(m)
    A = _phi(m + 1)
    samp_en = -np.log(A / B) if (A > 0 and B > 0) else np.nan
    return {f"{ch}_sample_entropy": float(samp_en)}

# correlations for all channel pairs within a window
def cross_channel(chunk: np.ndarray, channels: list) -> dict:
    feats = {}
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            x, y = chunk[:, i], chunk[:, j]
            r = 0.0 if (np.std(x) == 0 or np.std(y) == 0) else float(np.corrcoef(x, y)[0, 1])
            feats[f"corr_{channels[i]}_{channels[j]}"] = r
    return feats

# function to compute all features for every window and returns a df
def extract_features(windows: list) -> pd.DataFrame:
    n_ch = windows[0]["data"].shape[1]
    channels = IMU_CHANNELS[:n_ch]
    rows = []
    n = len(windows)

    for k, win in enumerate(windows):
        chunk = win["data"]
        feats: dict = {}
        for i, ch in enumerate(channels):
            x = chunk[:, i]
            feats.update(statistical_moments(x, ch))
            feats.update(temporal_structure(x, ch))
            feats.update(frequency_domain(x, ch))
            feats.update(entropy_features(x, ch))
        feats.update(cross_channel(chunk, channels))

        feats["driver"] = win["driver"]
        feats["behavior"] = win["behavior"]
        feats["road"] = win["road"]
        feats["trip_id"] = win["trip_id"]
        feats["window_id"] = win["window_id"]
        rows.append(feats)

        if (k + 1) % 1000 == 0:
            print(f"    features extracted: {k + 1:}/{n:}")

    df = pd.DataFrame(rows)
    meta_cols = ["driver", "behavior", "road", "trip_id", "window_id"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]
    print("-" * 60)
    print(f"feature extraction complete: {len(df):} windows x {len(feat_cols)} features")
    print("-" * 60)
    return df

#  ----  ----  ----  feature selection  ----  ----  ----

# first step: near-zero variance filter (NZV) and collinearity filter
# second step: MI ranking
# third step: Feature Selection
def select_features(
    feature_df: pd.DataFrame,
    near_zero_var_threshold: float = 0.001,
    collinearity_threshold: float = 0.95,) -> tuple:

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    meta_cols = ["driver", "behavior", "road", "trip_id", "window_id"]
    feat_cols = [c for c in feature_df.columns if c not in meta_cols]
    X = feature_df[feat_cols].copy()
    y_raw = feature_df["behavior"]
    label_map = {b: i for i, b in enumerate(BEHAVIOR_ORDER)}
    y = y_raw.map(label_map).fillna(0).astype(int)

    report = pd.DataFrame({"feature": feat_cols})
    report["kept_nzv"] = True
    report["kept_corr"] = True
    report["mi_score"] = np.nan

    # near zero variance filter
    variances = X.var(ddof=1)
    removed_nzv = [f for f in feat_cols if variances[f] < near_zero_var_threshold]
    report.loc[report["feature"].isin(removed_nzv), "kept_nzv"] = False
    X_nzv = X.drop(columns=removed_nzv)
    print(f"near zero variance filter removed {len(removed_nzv)} features ({len(X_nzv.columns)} left)")

    # collinearity filter
    corr_matrix = X_nzv.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    removed_corr = [c for c in upper.columns if any(upper[c] > collinearity_threshold)]
    report.loc[report["feature"].isin(removed_corr), "kept_corr"] = False
    X_filtered = X_nzv.drop(columns=removed_corr)
    print(f"collinearity filter removed {len(removed_corr)} features ({len(X_filtered.columns)} left)")
    print("-" * 60)


    # mutual information
    X_mi = X_filtered.fillna(X_filtered.median())
    mi_scores = mutual_info_classif(X_mi, y, discrete_features=False, random_state= RANDOM_STATE )
    mi_series = pd.Series(mi_scores, index=X_filtered.columns)

    for feat, score in mi_series.items():
        report.loc[report["feature"] == feat, "mi_score"] = score

    mi_ranked = mi_series.sort_values(ascending=False)
    print(f"\nfeatures by MI (Top 10):\n{mi_ranked.head(10).to_string()}")

    selected_df = pd.concat(
        [feature_df[meta_cols], X_filtered.fillna(X_filtered.median())], axis=1)

    report["kept_final"] = (
        report["kept_nzv"] & report["kept_corr"] &
        report["feature"].isin(X_filtered.columns))
    report = report.sort_values("mi_score", ascending=False).reset_index(drop=True)

    feat_out   = TABLE_DIR / "feature_matrix.csv"
    report_out = TABLE_DIR / "feature_selection.csv"
    selected_df.to_csv(feat_out, index=False)
    report.to_csv(report_out, index=False)
    return selected_df, report

# full preprocessing pipeline: segmentation → feature extraction → feature selection
def build_feature_matrix(
    all_accel: pd.DataFrame,
    run_selection: bool = True,) -> tuple:

    print("\n---- ---- Step 1: Segmentation ----  ----")
    windows = segment_trips(all_accel)

    print("\n---- ---- Step 2: Feature Extraction ----  ----")
    feature_df = extract_features(windows)

    if run_selection:
        print("\n---- ---- ---- Step 3: Feature Selection ---- ---- ----")
        feature_df, _ = select_features(feature_df)

    # creates an array y with integer labels for the behavior conditions,
    # and a meta DataFrame with driver, behavior, road, trip_id, window_id
    meta_cols = ["driver", "behavior", "road", "trip_id", "window_id"]
    feat_cols = [c for c in feature_df.columns if c not in meta_cols]
    meta = feature_df[meta_cols].reset_index(drop=True)
    X_df = feature_df[feat_cols].reset_index(drop=True)
    label_map = {b: i for i, b in enumerate(BEHAVIOR_ORDER)}
    y = meta["behavior"].map(label_map).fillna(0).astype(int).values

    print("\n---- ---- Step 4: PCA Scatter ----  ----")
    plot_pca_scatter(X_df, y)

    print("\n---- ---- Step 5a: Trip-Level Clustering k=3 ----  ----")
    trip_results = trip_level_clustering(X_df, y, meta)

    print("\n---- ---- Step 5b: Trip-Level Clustering k=6 ----  ----")
    trip_level_clustering_k6(X_df, y, meta)

    print("\n---- ---- Step 6: Within-Driver Clustering ----  ----")
    within_results = within_driver_clustering(X_df, y, meta)

    return X_df, y, meta, trip_results, within_results

# PCA scatter plot function
def plot_pca_scatter(X_df: pd.DataFrame, y: np.ndarray) -> None:

    COND_COLORS = {"NORMAL": "#4CAF50", "DROWSY": "#2196F3", "AGGRESSIVE": "#F44336",}

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    # 2D PCA for scatter
    pca2    = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca   = pca2.fit_transform(X_scaled)
    var_exp = pca2.explained_variance_ratio_

    # 1D PCA for PC1 means
    pca1  = PCA(n_components=1, random_state=RANDOM_STATE)
    X_pc1 = pca1.fit_transform(X_scaled)

    label_map = {b: i for i, b in enumerate(BEHAVIOR_ORDER)}
    pc1_means = {
        beh: float(X_pc1[y == label_map[beh], 0].mean())
        for beh in BEHAVIOR_ORDER}

    # Sample for plotting
    rng        = np.random.default_rng(373885)
    sample_idx = rng.choice(len(y), size=min(2500, len(y)), replace=False)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, beh in enumerate(BEHAVIOR_ORDER):
        mask = (y[sample_idx] == i)
        ax.scatter(
            X_pca[sample_idx][mask, 0],
            X_pca[sample_idx][mask, 1],
            c=COND_COLORS[beh], label=beh.title(),
            alpha=0.55, s=8, linewidths=0.2,
            edgecolors="white",)

    ax.set_xlabel(f"PC1 ({var_exp[0] * 100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_exp[1] * 100:.1f}% variance)", fontsize=10)

    ax.legend(
        fontsize=9, markerscale=4, framealpha=0.9,
        loc="lower right",
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.4,)

    textstr = "\n".join([
        "PC1 mean:",
        f"  Normal:     {pc1_means['NORMAL']:+.2f}",
        f"  Drowsy:     {pc1_means['DROWSY']:+.2f}",
        f"  Aggressive: {pc1_means['AGGRESSIVE']:+.2f}",])
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.8, edgecolor="#cccccc"))

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "pca_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nPCA scatter plot saved")

    # window report
    report = pd.DataFrame([{
        "PC1_var_explained": round(float(pca1.explained_variance_ratio_[0]), 4),
        "PC1_mean_NORMAL": round(pc1_means["NORMAL"], 4),
        "PC1_mean_DROWSY": round(pc1_means["DROWSY"], 4),
        "PC1_mean_AGGRESSIVE": round(pc1_means["AGGRESSIVE"], 4),
        "PC1_aligns_with_Normal": pc1_means["NORMAL"] == max(pc1_means.values()),}])

    report.to_csv(TABLE_DIR / "pca_window_report.csv", index=False)
    print(f"PCA window report saved")

# function to cluster at trip level at k=3 to test condition recovery
# Step 1. links metadata and features to aggregate all windows within individual trips
# Step 2. runs k-means (k=3) once on all 102 features and once on 10 component PCA reduced version
# Step 3. computes ARI and NMI against true behavior labels, and prints contingency tables
# Step 4. visualization by plot of 2D PCA and saving of results
def trip_level_clustering(
    X_df: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,) -> dict:

    trip_df  = pd.concat([meta, X_df], axis=1)
    trip_agg = (
        trip_df
        .groupby(["trip_id", "driver", "behavior"])[X_df.columns]
        .mean()
        .reset_index())

    label_map = {b: i for i, b in enumerate(BEHAVIOR_ORDER)}
    label_names = {i: b for i, b in enumerate(BEHAVIOR_ORDER)}
    y_trip = trip_agg["behavior"].map(label_map).values
    X_trip = trip_agg[X_df.columns].values

    print(f"\n")
    print(f"{len(trip_agg)} trips × {X_trip.shape[1]} features")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_trip)

    # k-means on all features
    km = KMeans(n_clusters=3, n_init=50, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    ari = adjusted_rand_score(y_trip, labels)
    nmi = normalized_mutual_info_score(y_trip, labels, average_method="arithmetic")

    print(f"k-means (all features): ARI={ari:.4f} NMI={nmi:.4f}")

    # k-means on PCA-reduced features
    n_components = min(10, len(trip_agg) - 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    km_pca = KMeans(n_clusters=3, n_init=50, random_state=RANDOM_STATE)
    labels_pca = km_pca.fit_predict(X_pca)
    ari_pca = adjusted_rand_score(y_trip, labels_pca)
    nmi_pca = normalized_mutual_info_score(y_trip, labels_pca, average_method="arithmetic")
    print(f"k-means (PCA {n_components} components): "f"ARI={ari_pca:.4f}  NMI={nmi_pca:.4f}")


    # Contingency tables
    cont_full = pd.crosstab(
        pd.Series(labels,     name="Cluster"),
        pd.Series([label_names[i] for i in y_trip], name="Condition"),)
    cont_pca = pd.crosstab(
        pd.Series(labels_pca, name="Cluster"),
        pd.Series([label_names[i] for i in y_trip], name="Condition"),)

    print(f"\nContingency Table (all features): \n{cont_full.to_string()}")
    print(f"\nContingency Table (PCA-reduced): \n{cont_pca.to_string()}")

    # 2D PCA for plotting
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X_scaled)
    var2 = pca2.explained_variance_ratio_

    COND_COLORS    = {"NORMAL": "#4CAF50", "DROWSY": "#2196F3", "AGGRESSIVE": "#F44336"}
    CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # left: true condition
    ax = axes[0]
    for i, beh in enumerate(BEHAVIOR_ORDER):
        mask = (y_trip == i)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COND_COLORS[beh], label=beh.title(),
                   s=80, edgecolors="white", linewidths=0.8, alpha=0.9)
    ax.set_xlabel(f"PC1 ({var2[0]*100:.1f}%)", fontsize=20)
    ax.set_ylabel(f"PC2 ({var2[1]*100:.1f}%)", fontsize=20)
    ax.legend(fontsize=16)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.set_facecolor("white")

    # right: cluster assignment
    ax = axes[1]
    for c in range(3):
        mask = (labels_pca == c)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=CLUSTER_COLORS[c], label=f"Cluster {c+1}",
                   s=80, edgecolors="white", linewidths=0.8,
                   alpha=0.9, marker="D")
    ax.set_xlabel(f"PC1 ({var2[0]*100:.1f}%)", fontsize=20)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.legend(fontsize=16)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.set_facecolor("white")

    plt.tight_layout()
    out = FIG_DIR / "clustering_k3.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save outputs
    report = pd.DataFrame([{
        "n_trips": len(trip_agg),
        "ARI_full": round(ari, 4),
        "NMI_full": round(nmi, 4),
        "ARI_pca": round(ari_pca, 4),
        "NMI_pca": round(nmi_pca, 4),
        "pca_components": n_components,}])

    report.to_csv(TABLE_DIR / "clustering_k3.csv", index=False)
    cont_full.to_csv(TABLE_DIR / "cont_tab_full_k3.csv")
    cont_pca.to_csv(TABLE_DIR  / "cont_tab_pca_k3.csv")

    return { "ari_full": ari,
             "nmi_full": nmi,
             "ari_pca": ari_pca,
             "nmi_pca": nmi_pca,
             "labels_full": labels,
             "labels_pca": labels_pca,
             "X_2d": X_2d,
             "y_trip": y_trip,
             "trip_agg": trip_agg,}

# function to cluster at trip level at k=6 to test driver identity recovery
# Step 1. links metadata and features to aggregate all windows within individual trips
# Step 2. runs k-means (k=6) on 10 component PCA reduced version
# Step 3. computes ARI and NMI against both driver identity and condition labels, and prints contingency tables
# Step 4. visualization by plot of 2D PCA and saving of results
def trip_level_clustering_k6(
    X_df: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,) -> None:

    # aggregate to trip level
    trip_df  = pd.concat([meta, X_df], axis=1)
    trip_agg = (
        trip_df
        .groupby(["trip_id", "driver", "behavior"])[X_df.columns]
        .mean()
        .reset_index())

    X_trip = trip_agg[X_df.columns].values
    y_driver = trip_agg["driver"].values
    y_cond = trip_agg["behavior"].map({b: i for i, b in enumerate(BEHAVIOR_ORDER)}).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trip)

    # PCA reduce then cluster k=6
    pca = PCA(n_components=10, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    km = KMeans(n_clusters=6, n_init=50, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_pca)

    # ARI against driver identity and condition
    driver_map = {d: i for i, d in enumerate(sorted(set(y_driver)))}
    y_drv_int  = np.array([driver_map[d] for d in y_driver])
    ari_driver = adjusted_rand_score(y_drv_int, labels)
    ari_cond   = adjusted_rand_score(y_cond,    labels)

    print(f"\nARI (driver identity): {ari_driver:.4f}")
    print(f"ARI (condition): {ari_cond:.4f}")

    # 2D PCA for plotting
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca2.fit_transform(X_scaled)
    var2 = pca2.explained_variance_ratio_

    DRIVER_COLORS = {"D1": "#e41a1c", "D2": "#377eb8", "D3": "#4daf4a",
                     "D4": "#ff7f00", "D5": "#984ea3", "D6": "#a65628",}
    CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a",
                      "#ff7f00", "#984ea3", "#a65628"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # left: true driver identity
    ax = axes[0]
    for driver, color in DRIVER_COLORS.items():
        mask = (y_driver == driver)
        if mask.sum() == 0:
            continue
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, label=driver,
            s=80, edgecolors="white", linewidths=0.8, alpha=0.9)
    ax.set_xlabel(f"PC1 ({var2[0]*100:.1f}%)", fontsize=20)
    ax.set_ylabel(f"PC2 ({var2[1]*100:.1f}%)", fontsize=20)
    ax.legend(fontsize=16, title="Driver")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.set_facecolor("white")

    # right: k=6 cluster assignments
    ax = axes[1]
    for c in range(6):
        mask = (labels == c)
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=CLUSTER_COLORS[c], label=f"Cluster {c+1}",
            s=80, edgecolors="white", linewidths=0.8,
            alpha=0.9, marker="D")
    ax.set_xlabel(f"PC1 ({var2[0]*100:.1f}%)", fontsize=20)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.legend(fontsize=16, title="Cluster")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.set_facecolor("white")

    plt.tight_layout()
    out = FIG_DIR / "clustering_k6.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame([{ "ARI_vs_driver": round(ari_driver, 4),
                    "ARI_vs_condition": round(ari_cond, 4),
                    }]).to_csv(TABLE_DIR / "clustering_k6.csv", index=False)

    # ── Step 4: within driver clustering ─────────────────────────────────────

# function to cluster at window level separately within each driver (k=3)
# Step 1. iterating over drivers and subset windows from each driver
# Step 2. run k-means (k=3) for each driver
# step 3. compute ARI and NMI against true behavior labels
# step 4. saves results and plot figure
def within_driver_clustering(
    X_df: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,) -> dict:

    rows = []
    for driver in sorted(meta["driver"].unique()):
        mask     = (meta["driver"] == driver).values
        X_driver = X_df.values[mask]
        y_driver = y[mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_driver)

        km = KMeans(n_clusters=3, n_init=50, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)

        ari = adjusted_rand_score(y_driver, labels)
        nmi = normalized_mutual_info_score(y_driver, labels, average_method="arithmetic")

        n_windows = len(y_driver)
        print(f"    {driver}: ARI={ari:.4f}  NMI={nmi:.4f} "f"(n={n_windows} windows)")

        rows.append({"Driver": driver,
                     "N_windows": n_windows,
                     "ARI": round(ari, 4),
                     "NMI": round(nmi, 4),})

    results_df = pd.DataFrame(rows)

    print(f"\nMean ARI across drivers: "f"{results_df['ARI'].mean():.4f} "f"(SD={results_df['ARI'].std():.4f})")

    out = TABLE_DIR / "within_driver_clustering.csv"
    results_df.to_csv(out, index=False)
    plot_within_driver_clustering(results_df)

    return {"results_df": results_df,
            "mean_ari": results_df["ARI"].mean(),
            "sd_ari": results_df["ARI"].std(),}


# plot ARI per driver
def plot_within_driver_clustering(results_df: pd.DataFrame):

    DRIVER_COLORS = {"D1": "#e41a1c", "D2": "#377eb8", "D3": "#4daf4a",
                     "D4": "#984ea3", "D5": "#ff7f00", "D6": "#a65628",}

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        results_df["Driver"],
        results_df["ARI"],
        color=[DRIVER_COLORS.get(d, "grey") for d in results_df["Driver"]],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
        width=0.5,)

    # Cross-driver ARI line
    ax.axhline(0.006, color="black", linestyle="--",
               linewidth=1.2, label="Cross-driver ARI (trip-level)")

    # mean within-driver ARI
    mean_ari = results_df["ARI"].mean()
    ax.axhline(mean_ari, color="grey", linestyle="-",
               linewidth=1.2, label=f"Mean within-driver ARI ({mean_ari:.3f})")

    # labels on bars
    for bar, ari in zip(bars, results_df["ARI"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{ari:.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Driver", fontsize=12)
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=12)
    ax.set_title("Within-Driver Clustering vs Cross-Driver Baseline\n"
                 "k-means k=3, window level",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, min(1.0, results_df["ARI"].max() + 0.15))
    ax.legend(fontsize=10, framealpha=0.9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "within_driver_clustering.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# combined plot for ACF and ARI
def plot_acf_ari(
    window_scores: pd.DataFrame,
    within_driver_results: dict,
    output_dir: Path = FIG_DIR,) -> None:


    # ACF calculation
    ws_xgb = window_scores[window_scores["model"] == "XGBoost"]
    acf_per_trip = []
    for trip_id, grp in ws_xgb.groupby("trip_id"):
        vals = grp.sort_values("window_id")["p_normal"].values
        if len(vals) >= 10:
            a = acf(vals, nlags=10, fft=True)
            acf_per_trip.append(a)

    mean_acf = np.mean(acf_per_trip, axis=0)
    lags = np.arange(1, 11)
    acf_vals = mean_acf[1:]

    # ARI
    drivers = sorted(within_driver_results.keys())
    ari_vals = [float(within_driver_results[d]) for d in drivers]
    mean_ari = np.mean(ari_vals)
    baseline = 0.006

    DRIVER_COLORS = {"D1": "#e41a1c", "D2": "#377eb8", "D3": "#4daf4a",
                     "D4": "#ff7f00", "D5": "#984ea3", "D6": "#a65628",}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("white")

    # left: ACF
    ax = axes[0]
    ax.set_facecolor("white")
    ax.bar(lags, acf_vals, color="steelblue", alpha=0.75, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag", fontsize=15)
    ax.set_ylabel("Mean ACF", fontsize=15)
    ax.set_xticks(lags)
    ax.tick_params(labelsize=13)
    ax.set_ylim(0, 0.30)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # right: ARI per driver
    ax = axes[1]
    ax.set_facecolor("white")
    x    = np.arange(len(drivers))
    bars = ax.bar(
        x, ari_vals, width=0.6,
        color=[DRIVER_COLORS[d] for d in drivers],
        alpha=0.8)
    for bar, val in zip(bars, ari_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=12)

    ax.axhline(baseline, color="black", linewidth=1.2,
               linestyle="--", label=f"Cross-driver ARI ({baseline:.3f})")
    ax.axhline(mean_ari, color="grey", linewidth=1.2,
               linestyle="-", label=f"Mean within-driver ARI ({mean_ari:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(drivers, fontsize=13)
    ax.set_xlabel("Driver", fontsize=15)
    ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_ylim(0, 0.08)
    ax.legend(fontsize=12, framealpha=0.9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.95,
                        bottom=0.14, wspace=0.28)
    out = output_dir / "acf_ari.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)




# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
#  ###  SECTION 3: MODELING
# ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----

LABEL_MAP = {b: i for i, b in enumerate(BEHAVIOR_ORDER)}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# sklearn implementation of logistic regression, random forest and XGB
def build_models() -> dict:
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, solver="lbfgs",)),])
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,)),])
    xgb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1,)),])
    return {"LogisticRegression": lr, "RandomForest": rf, "XGBoost": xgb}

# leave one driver out cross validation function
def lodo_cv(
    model_pipeline,
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    model_name: str,) -> dict:

    drivers = meta["driver"].unique()
    all_proba = np.zeros((len(y), 3))
    all_preds = np.zeros(len(y), dtype=int)
    fold_results = []

    print(f"\n{'--- '*16}\n  {model_name} — Leave-One-Driver-Out CV\n{'--- '*16}")

    for held_out in sorted(drivers):
        train_idx = meta.index[meta["driver"] != held_out].tolist()
        test_idx  = meta.index[meta["driver"] == held_out].tolist()

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_pipeline.fit(X_train, y_train)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
        cal_local, eval_local = next(sss.split(np.zeros(len(test_idx)), y[test_idx]))
        cal_idx  = [test_idx[i] for i in cal_local]

        calibrated = CalibratedClassifierCV(model_pipeline, cv=None, method="isotonic", ensemble=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated.fit(X[cal_idx], y[cal_idx])

        proba_full = calibrated.predict_proba(X[test_idx])
        preds_full = np.argmax(proba_full, axis=1)
        all_proba[test_idx] = proba_full
        all_preds[test_idx] = preds_full

        acc = accuracy_score(y[test_idx], preds_full)
        f1 = f1_score(y[test_idx], preds_full, average="macro", zero_division=0)
        f1pc = f1_score(y[test_idx], preds_full, average=None, zero_division=0)
        ece = expected_calibration_error(
            (y[test_idx] == NORMAL_IDX).astype(int), proba_full[:, NORMAL_IDX])

        fold_results.append({
            "held_out_driver": held_out,
            "n_test_windows": len(test_idx),
            "accuracy": acc,
            "macro_f1": f1,
            "f1_normal": f1pc[NORMAL_IDX] if len(f1pc) > NORMAL_IDX else np.nan,
            "f1_drowsy": f1pc[1] if len(f1pc) > 1 else np.nan,
            "f1_aggressive": f1pc[2] if len(f1pc) > 2 else np.nan,
            "ece_normal": ece,})
        print(f"  held out {held_out}: acc={acc:.3f}  macro F1={f1:.3f}  ECE(Normal)={ece:.3f}")

    fold_df = pd.DataFrame(fold_results)
    print(f"\n mean across folds:")
    print(f" accuracy: {fold_df['accuracy'].mean():.3f} (SD={fold_df['accuracy'].std():.3f})")
    print(f" macro F1: {fold_df['macro_f1'].mean():.3f} (SD={fold_df['macro_f1'].std():.3f})")
    print(f" ECE(Normal): {fold_df['ece_normal'].mean():.3f} (SD={fold_df['ece_normal'].std():.3f})")

    window_scores = meta.copy().reset_index(drop=True)
    window_scores["p_normal"] = all_proba[:, NORMAL_IDX]
    window_scores["p_drowsy"] = all_proba[:, 1]
    window_scores["p_aggressive"] = all_proba[:, 2]
    window_scores["pred_label"] = [INV_LABEL_MAP[p] for p in all_preds]
    window_scores["true_label"] = [INV_LABEL_MAP[i] for i in y]
    window_scores["model"] = model_name

    return {"fold_results": fold_df,
            "all_proba": all_proba,
            "all_preds": all_preds,
            "all_true": y,
            "window_scores": window_scores,}

# function to aggregate window-level P(Normal) to trip and person × condition level
# by taking the mean across windows, and also computing SD and count of windows
# for each trip and person × condition
def aggregate_scores(window_scores: pd.DataFrame) -> tuple:
    trip_scores = (
        window_scores
        .groupby(["trip_id", "driver", "behavior", "road", "model"])
        .agg(mean_p_normal = ("p_normal", "mean"),
             sd_p_normal = ("p_normal", "std"),
             n_windows = ("p_normal", "count"),).reset_index())

    person_scores = (trip_scores.groupby(["driver", "behavior", "model"]).agg(
            mean_p_normal = ("mean_p_normal", "mean"),
            sd_p_normal = ("mean_p_normal", "std"),
            n_trips = ("trip_id", "count"),).reset_index())
    return trip_scores, person_scores

# function to run the full modeling pipeline:
# LODO CV → calibration → scoring for all three models
# returns a dict of results
def run_modeling_pipeline(
    X_df: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,) -> dict:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X = X_df.values.astype(float)
    models = build_models()
    results = {}

    for name, pipeline in models.items():
        results[name] = lodo_cv(pipeline, X, y, meta.reset_index(drop=True), name)

    summary = build_summary_table(results)
    print("\n" + "="*100)
    print("Model comparison summary (mean ± SD across LODO folds)")
    print("="*100)
    print(summary.to_string(index=False))
    summary.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    results["summary_table"] = summary

    cv_only = {k: v for k, v in results.items() if k != "summary_table"}
    plot_calibration_curves(cv_only)
    plot_lodo_performance(cv_only)

    all_window_scores = pd.concat(
        [res["window_scores"] for res in cv_only.values()], ignore_index=True)
    all_window_scores.to_csv(TABLE_DIR / "window_scores.csv", index=False)

    trip_scores_all, person_scores_all = [], []
    for name in cv_only:
        ws = cv_only[name]["window_scores"]
        trip_s, pers_s = aggregate_scores(ws)
        trip_scores_all.append(trip_s)
        person_scores_all.append(pers_s)

    trip_scores = pd.concat(trip_scores_all, ignore_index=True)
    person_scores = pd.concat(person_scores_all, ignore_index=True)

    score_distributions(person_scores)
    trip_scores.to_csv(TABLE_DIR / "trip_scores.csv", index=False)
    person_scores.to_csv(TABLE_DIR / "person_scores.csv", index=False)

    results["trip_scores"]   = trip_scores
    results["person_scores"] = person_scores
    return results