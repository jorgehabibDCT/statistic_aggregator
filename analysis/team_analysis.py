import pandas as pd


def _is_usable_sample(vs_sample_size, min_sample):
    return vs_sample_size >= min_sample


def build_player_summary_row(player_name, stat_results, profile, min_sample):
    if stat_results.empty:
        return None

    best_row = stat_results.sort_values("delta", ascending=False).iloc[0]
    worst_row = stat_results.sort_values("delta", ascending=True).iloc[0]
    stable_row = stat_results.sort_values("volatility_score", ascending=True).iloc[0]
    overall_confidence = int(stat_results["confidence"].mean())

    usable_sample = _is_usable_sample(int(best_row["vs_sample_size"]), min_sample)
    return {
        "player": player_name,
        "best_elevated_stat": best_row["stat"],
        "best_elevated_delta": float(best_row["delta"]),
        "strongest_negative_stat": worst_row["stat"],
        "most_stable_stat": stable_row["stat"],
        "stability_label": stable_row["stability_label"],
        "overall_confidence": overall_confidence,
        "profile": profile,
        "vs_sample_size": int(best_row["vs_sample_size"]),
        "opponent_sample_adequate": "yes" if usable_sample else "low",
    }


def build_signal_rows(player_name, stat_results, profile, min_sample):
    signal_rows = []
    for _, row in stat_results.iterrows():
        signal_rows.append(
            {
                "player": player_name,
                "stat": row["stat"],
                "delta": float(row["delta"]),
                "split_label": row["split_label"],
                "stability_label": row["stability_label"],
                "confidence": int(row["confidence"]),
                "profile": profile,
                "vs_sample_size": int(row["vs_sample_size"]),
                "opponent_sample_adequate": int(row["vs_sample_size"]) >= min_sample,
                "volatility_score": float(row["volatility_score"]),
            }
        )
    return signal_rows


def compute_signal_score(signal_row):
    split_strength = min(abs(signal_row["delta"]) / 8.0, 1.0)
    stability_factor = 1.0 - min(signal_row["volatility_score"], 1.0)
    confidence_factor = min(signal_row["confidence"] / 100.0, 1.0)
    sample_factor = min(signal_row["vs_sample_size"] / 10.0, 1.0)
    adequacy_bonus = 0.1 if signal_row["opponent_sample_adequate"] else -0.1
    noise_penalty = 0.0
    if signal_row["stat"] in {"STL", "BLK", "BS"} and signal_row["volatility_score"] > 0.5:
        noise_penalty = 0.08

    score = (0.35 * split_strength) + (0.3 * stability_factor) + (0.25 * confidence_factor) + (0.1 * sample_factor) + adequacy_bonus - noise_penalty
    return round(max(0.0, min(1.0, score)) * 100, 1)


def finalize_signal_board(signal_rows):
    if not signal_rows:
        return pd.DataFrame()

    df = pd.DataFrame(signal_rows)
    df["signal_score"] = df.apply(compute_signal_score, axis=1)
    return df.sort_values(["signal_score", "delta", "confidence"], ascending=[False, False, False]).reset_index(drop=True)
