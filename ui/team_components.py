import streamlit as st

from ui.components import format_stat_label


def render_top_signals_panel(signal_df, top_n=12):
    st.subheader("Top Signals")
    if signal_df.empty:
        st.info("No signals met current filters.")
        return

    top = signal_df.head(top_n)
    display_df = top.copy()
    display_df["stat"] = display_df["stat"].apply(format_stat_label)
    st.dataframe(
        display_df[
            [
                "player",
                "stat",
                "delta",
                "stability_label",
                "confidence",
                "signal_score",
                "profile",
            ]
        ],
        use_container_width=True,
    )


def render_team_summary_table(summary_df):
    st.subheader("Team Summary")
    if summary_df.empty:
        st.info("No team summary rows available.")
        return

    display_df = summary_df.copy()
    for col in ["best_elevated_stat", "strongest_negative_stat", "most_stable_stat"]:
        display_df[col] = display_df[col].apply(format_stat_label)
    st.dataframe(
        display_df[
            [
                "player",
                "best_elevated_stat",
                "best_elevated_delta",
                "strongest_negative_stat",
                "most_stable_stat",
                "stability_label",
                "overall_confidence",
                "profile",
                "vs_sample_size",
                "opponent_sample_adequate",
            ]
        ],
        use_container_width=True,
    )
