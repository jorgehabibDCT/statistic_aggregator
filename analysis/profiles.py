def infer_player_profile(stat_results):
    by_stat = {row["stat"]: row for _, row in stat_results.iterrows()}
    pts = by_stat.get("PTS", {})
    reb = by_stat.get("REB", {})
    ast = by_stat.get("AST", {})

    pts_delta = pts.get("delta", 0)
    reb_delta = reb.get("delta", 0)
    ast_delta = ast.get("delta", 0)

    high_vol_count = int((stat_results["stability_label"] == "high volatility").sum())
    all_main_negative = pts_delta < 0 and reb_delta < 0 and ast_delta < 0

    if high_vol_count >= 3:
        return "volatile / noisy sample"
    if all_main_negative:
        return "suppressed matchup"
    if pts_delta > 1.5 and ast_delta < -0.5:
        return "scoring role increase"
    if ast_delta > 1.0 or ast.get("stability_label") in {"very stable", "fairly stable"}:
        return "facilitator profile"
    if reb_delta > 1.0 and pts_delta < 0:
        return "rebound-heavy matchup"
    if (reb_delta + ast_delta) > 1.5 and pts_delta <= 0.5:
        return "peripheral-driven matchup"
    return "balanced / neutral profile"
