import numpy as np
import pandas as pd

from analysis.classifiers import classify_split, classify_stability


def add_combo_stats(games):
    df = games.copy()
    for col in ["PTS", "REB", "AST"]:
        if col not in df.columns:
            df[col] = 0
    df["PR"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0) + pd.to_numeric(df["REB"], errors="coerce").fillna(0)
    df["PA"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0) + pd.to_numeric(df["AST"], errors="coerce").fillna(0)
    df["RA"] = pd.to_numeric(df["REB"], errors="coerce").fillna(0) + pd.to_numeric(df["AST"], errors="coerce").fillna(0)
    df["PRA"] = df["PR"] + pd.to_numeric(df["AST"], errors="coerce").fillna(0)
    return df


def get_stat_series(games, stat_name):
    if stat_name not in games.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(games[stat_name], errors="coerce").dropna()


def compute_volatility_metrics(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {
            "std_dev": 0.0,
            "range_value": 0.0,
            "cv": 0.0,
            "near_mean_ratio": 0.0,
            "outlier_ratio": 0.0,
            "volatility_score": 1.0,
        }

    mean = float(np.mean(arr))
    std_dev = float(np.std(arr, ddof=0))
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    range_value = max_value - min_value

    cv = std_dev / abs(mean) if mean != 0 else std_dev
    near_mean_ratio = float(np.mean(np.abs(arr - mean) <= std_dev)) if std_dev > 0 else 1.0
    outlier_ratio = float(np.mean(np.abs(arr - mean) > 2 * std_dev)) if std_dev > 0 else 0.0

    # Weighted score where lower is more stable.
    score = (0.45 * min(cv, 1.5) / 1.5) + (0.25 * min(range_value / max(abs(mean), 1.0), 2.5) / 2.5) + (0.2 * (1 - near_mean_ratio)) + (0.1 * outlier_ratio)
    return {
        "std_dev": std_dev,
        "range_value": range_value,
        "cv": float(cv),
        "near_mean_ratio": near_mean_ratio,
        "outlier_ratio": outlier_ratio,
        "volatility_score": float(np.clip(score, 0.0, 1.0)),
    }


def compute_stat_summary(vs_games, overall_games, stat_name, edge_thresholds, stability_thresholds):
    vs_series = get_stat_series(vs_games, stat_name)
    overall_series = get_stat_series(overall_games, stat_name)

    vs_avg = float(vs_series.mean()) if not vs_series.empty else 0.0
    overall_avg = float(overall_series.mean()) if not overall_series.empty else 0.0
    delta = vs_avg - overall_avg
    pct_change = (delta / overall_avg) if overall_avg != 0 else 0.0

    vol = compute_volatility_metrics(vs_series.tolist())
    split_label = classify_split(delta, pct_change, edge_thresholds)
    stability_label = classify_stability(vol["volatility_score"], stability_thresholds)

    return {
        "stat": stat_name,
        "vs_avg": round(vs_avg, 2),
        "overall_avg": round(overall_avg, 2),
        "delta": round(delta, 2),
        "pct_change": round(pct_change * 100, 1),
        "min": round(float(vs_series.min()) if not vs_series.empty else 0.0, 2),
        "max": round(float(vs_series.max()) if not vs_series.empty else 0.0, 2),
        "median": round(float(vs_series.median()) if not vs_series.empty else 0.0, 2),
        "std_dev": round(vol["std_dev"], 2),
        "cv": round(vol["cv"], 3),
        "range_value": round(vol["range_value"], 2),
        "vs_sample_size": int(vs_series.size),
        "overall_sample_size": int(overall_series.size),
        "split_label": split_label,
        "stability_label": stability_label,
        "volatility_score": round(vol["volatility_score"], 3),
        "near_mean_ratio": round(vol["near_mean_ratio"], 3),
        "outlier_ratio": round(vol["outlier_ratio"], 3),
    }


def compute_confidence_score(result):
    sample_factor = min(result["vs_sample_size"] / 10.0, 1.0)
    split_strength = min(abs(result["pct_change"]) / 25.0, 1.0)
    stability_factor = 1.0 - min(result["volatility_score"], 1.0)
    outlier_penalty = min(result["outlier_ratio"], 0.5)

    score = (40 * stability_factor) + (25 * sample_factor) + (25 * split_strength) - (20 * outlier_penalty)
    return max(0, min(100, int(round(score))))


def run_comparison_engine(vs_games, overall_games, stats, edge_thresholds, stability_thresholds):
    results = []
    for stat in stats:
        row = compute_stat_summary(vs_games, overall_games, stat, edge_thresholds, stability_thresholds)
        row["confidence"] = compute_confidence_score(row)
        results.append(row)

    return pd.DataFrame(results)
