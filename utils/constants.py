BASE_STATS = ["PTS", "REB", "AST", "FG3M", "FG3A"]
COMBO_STATS = ["PR", "PA", "RA", "PRA"]
DEFAULT_STATS = BASE_STATS + COMBO_STATS

SAMPLE_OPTIONS = [5, 7, 10]
DEFAULT_VS_SAMPLE = 7
DEFAULT_OVERALL_SAMPLE = 7

LOW_SAMPLE_THRESHOLD = 5

EDGE_THRESHOLDS = {
    "strong_positive_pct": 0.15,
    "mild_positive_pct": 0.05,
    "mild_negative_pct": -0.05,
    "strong_negative_pct": -0.15,
}

STABILITY_THRESHOLDS = {
    "very_stable_max": 0.18,
    "fairly_stable_max": 0.32,
    "moderate_max": 0.50,
}
