import streamlit as st


def render_summary_cards(best_row, worst_row, stable_row, volatile_row, profile, overall_confidence):
    st.subheader("Insight Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Elevated Stat", f"{best_row['stat']} ({best_row['delta']:+.1f})")
    col2.metric("Most Suppressed Stat", f"{worst_row['stat']} ({worst_row['delta']:+.1f})")
    col3.metric("Most Stable Stat", f"{stable_row['stat']} ({stable_row['stability_label']})")
    col4.metric("Most Volatile Stat", f"{volatile_row['stat']} ({volatile_row['stability_label']})")

    st.markdown(f"**Profile:** `{profile}`")
    st.progress(min(max(overall_confidence, 0), 100), text=f"Overall Confidence: {overall_confidence}/100")


def render_comparison_table(stat_results):
    st.subheader("Split Comparison")
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
    st.dataframe(stat_results[display_cols], use_container_width=True)


def render_low_sample_warning(vs_sample_size, threshold):
    if vs_sample_size < threshold:
        st.warning(
            f"Low opponent sample: only {vs_sample_size} games vs this opponent. Treat conclusions as directional."
        )


def render_summary_text(text):
    st.subheader("Analyst Summary")
    st.info(text)
