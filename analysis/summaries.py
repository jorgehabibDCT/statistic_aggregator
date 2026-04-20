def _pick_rows(stat_results):
    best = stat_results.sort_values("delta", ascending=False).iloc[0]
    worst = stat_results.sort_values("delta", ascending=True).iloc[0]
    stable = stat_results.sort_values("volatility_score", ascending=True).iloc[0]
    volatile = stat_results.sort_values("volatility_score", ascending=False).iloc[0]
    return best, worst, stable, volatile


def build_natural_language_summary(stat_results, profile, overall_confidence):
    if stat_results.empty:
        return "No data available to generate matchup insights."

    best, worst, stable, volatile = _pick_rows(stat_results)
    sample_quality = "clean" if (stat_results["stability_label"].isin(["very stable", "fairly stable"]).mean() >= 0.6) else "noisy"

    confidence_tier = "high" if overall_confidence >= 70 else "medium" if overall_confidence >= 45 else "low"

    return (
        f"This matchup shows the strongest lift in {best['stat']} ({best['delta']:+.1f}, {best['pct_change']:+.1f}%) "
        f"and the biggest suppression in {worst['stat']} ({worst['delta']:+.1f}, {worst['pct_change']:+.1f}%). "
        f"The most trustworthy signal is {stable['stat']} ({stable['stability_label']}), while {volatile['stat']} looks least reliable "
        f"due to {volatile['stability_label']}. Overall this reads as a {profile} with a {sample_quality} sample and {confidence_tier} confidence."
    )
