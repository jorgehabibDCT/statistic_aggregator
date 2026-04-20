def classify_split(delta, pct_change, thresholds):
    if pct_change >= thresholds["strong_positive_pct"]:
        return "strong positive split"
    if pct_change >= thresholds["mild_positive_pct"]:
        return "mild positive split"
    if pct_change <= thresholds["strong_negative_pct"]:
        return "strong negative split"
    if pct_change <= thresholds["mild_negative_pct"]:
        return "mild negative split"
    return "neutral"


def classify_stability(volatility_score, thresholds):
    if volatility_score <= thresholds["very_stable_max"]:
        return "very stable"
    if volatility_score <= thresholds["fairly_stable_max"]:
        return "fairly stable"
    if volatility_score <= thresholds["moderate_max"]:
        return "moderate volatility"
    return "high volatility"
