import streamlit as st


def format_stat_label(stat_name):
    return "B+S" if stat_name == "BS" else stat_name


def render_summary_cards(best_row, worst_row, stable_row, volatile_row, profile, overall_confidence):
    st.subheader("Insight Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Elevated Stat", f"{format_stat_label(best_row['stat'])} ({best_row['delta']:+.1f})")
    col2.metric("Most Suppressed Stat", f"{format_stat_label(worst_row['stat'])} ({worst_row['delta']:+.1f})")
    col3.metric("Most Stable Stat", f"{format_stat_label(stable_row['stat'])} ({stable_row['stability_label']})")
    col4.metric("Most Volatile Stat", f"{format_stat_label(volatile_row['stat'])} ({volatile_row['stability_label']})")

    st.markdown(f"**Profile:** `{profile}`")
    st.progress(min(max(overall_confidence, 0), 100), text=f"Overall Confidence: {overall_confidence}/100")


def render_comparison_table(stat_results):
    st.subheader("Split Comparison")
    display_df = stat_results.copy()
    display_df["stat"] = display_df["stat"].apply(format_stat_label)
    display_cols = [
        "stat",
        "vs_avg",
        "overall_avg",
        "delta",
        "pct_change",
        "split_label",
        "stability_label",
        "std_dev",
        "cv",
        "confidence",
        "vs_sample_size",
    ]
    st.dataframe(display_df[display_cols], width="stretch")


def render_low_sample_warning(vs_sample_size, threshold):
    if vs_sample_size < threshold:
        st.warning(
            f"Low opponent sample: only {vs_sample_size} games vs this opponent. Treat conclusions as directional."
        )


def render_summary_text(text):
    st.subheader("Analyst Summary")
    st.info(text)


def render_hit_rate_panel(stat_name, threshold, vs_result, overall_result, benchmark_thresholds):
    st.subheader("Hit-Rate Analysis")
    st.caption("Generic threshold analysis for matchup trend context.")
    st.markdown(f"**Selected:** `{format_stat_label(stat_name)}` at threshold `>= {threshold}`")

    col1, col2 = st.columns(2)
    col1.metric(
        "Vs Opponent Hit Rate",
        f"{vs_result['hit_rate_pct']}%",
        f"{vs_result['hits']}/{vs_result['attempts']}",
    )
    col2.metric(
        "Overall Hit Rate",
        f"{overall_result['hit_rate_pct']}%",
        f"{overall_result['hits']}/{overall_result['attempts']}",
    )

    if benchmark_thresholds:
        st.caption(f"Auto benchmarks for {format_stat_label(stat_name)}: {', '.join(str(v) for v in benchmark_thresholds)}")
