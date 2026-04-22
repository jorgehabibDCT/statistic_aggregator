import pandas as pd
import streamlit as st

from ui.components import format_stat_label


def render_compact_comparison_charts(stat_results, stats=("PTS", "REB", "AST")):
    st.subheader("Compact Stat Comparison")
    chart_cols = st.columns(len(stats))

    for idx, stat in enumerate(stats):
        row = stat_results[stat_results["stat"] == stat]
        if row.empty:
            chart_cols[idx].caption(f"{stat}: no data")
            continue

        chart_df = pd.DataFrame(
            {
                "split": ["vs_opponent", "overall"],
                "value": [float(row.iloc[0]["vs_avg"]), float(row.iloc[0]["overall_avg"])],
            }
        ).set_index("split")
        chart_cols[idx].caption(format_stat_label(stat))
        chart_cols[idx].bar_chart(chart_df, y="value")
