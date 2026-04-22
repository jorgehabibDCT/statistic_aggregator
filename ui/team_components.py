import streamlit as st

from analysis.metrics import compute_hit_rate
from ui.components import format_stat_label


def render_top_signals_panel(signal_df, top_n=12, theme_name="neon", player_context_map=None, section_key="signals"):
    st.subheader("Top Signals")
    if signal_df.empty:
        st.info("No signals met current filters.")
        return
    if player_context_map is None:
        player_context_map = {}

    top = signal_df.head(top_n)
    display_df = top.copy()
    display_df["stat"] = display_df["stat"].apply(format_stat_label)

    palette = {
        "neon": {"accent": "#00f5ff", "accent2": "#ff2bd6"},
        "retro": {"accent": "#00ecff", "accent2": "#ff39d4"},
        "mono": {"accent": "#00d8ff", "accent2": "#d455ff"},
    }
    colors = palette.get(theme_name, palette["neon"])

    for row_idx, row in display_df.iterrows():
        signal_score = float(row.get("signal_score", 0))
        intensity = max(0.2, min(1.0, signal_score / 100.0))
        border_alpha = 0.25 + (0.55 * intensity)
        glow_alpha = 0.15 + (0.3 * intensity)
        row_key = f"{section_key}_{row_idx}_{row['player']}_{row['stat']}".replace(" ", "_")
        threshold_key = f"{row_key}_threshold"
        form_key = f"{row_key}_form"
        result_key = f"{row_key}_result"
        st.markdown(
            f"""
            <div style="
                border: 1px solid rgba(255,255,255,{border_alpha:.2f});
                border-left: 6px solid {colors['accent']};
                box-shadow: 0 0 8px rgba(0,0,0,0.35), 0 0 10px rgba(255,255,255,{glow_alpha:.2f});
                border-radius: 1px;
                padding: 10px 12px;
                margin-bottom: 8px;
                background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            ">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
                <div style="font-weight:700;font-size:1.02rem;">{row['player']}</div>
                <div style="font-size:0.82rem;opacity:0.85;">{row['profile']}</div>
              </div>
              <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:12px;margin-top:6px;">
                <div>
                  <div style="font-size:0.78rem;opacity:0.85;">Stat</div>
                  <div style="font-size:1.15rem;font-weight:800;color:{colors['accent']};">{row['stat']}</div>
                </div>
                <div>
                  <div style="font-size:0.78rem;opacity:0.85;">Delta</div>
                  <div style="font-size:1.2rem;font-weight:900;color:{colors['accent2']};">{row['delta']:+.1f}</div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:0.78rem;opacity:0.85;">Confidence</div>
                  <div style="font-size:1.0rem;font-weight:700;">{int(row['confidence'])}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        context = player_context_map.get(row["player"])
        if context:
            with st.form(form_key, clear_on_submit=False):
                st.number_input(
                    f"Line for {row['player']} {row['stat']} (>=)",
                    min_value=0.0,
                    step=0.5,
                    value=float(st.session_state.get(threshold_key, 10.0)),
                    key=threshold_key,
                    format="%.1f",
                )
                submitted = st.form_submit_button("Check Hit Rate")

            if submitted:
                threshold_value = float(st.session_state[threshold_key])
                vs_result = compute_hit_rate(context["vs_for_analysis"], row["stat"], threshold_value)
                overall_result = compute_hit_rate(context["overall_for_analysis"], row["stat"], threshold_value)
                st.session_state[result_key] = {
                    "threshold": threshold_value,
                    "vs": vs_result,
                    "overall": overall_result,
                }

            result = st.session_state.get(result_key)
            if result:
                st.caption(
                    f"Line {result['threshold']:.1f} | "
                    f"vs opp: {result['vs']['hits']}/{result['vs']['attempts']} ({result['vs']['hit_rate_pct']}%) | "
                    f"overall: {result['overall']['hits']}/{result['overall']['attempts']} ({result['overall']['hit_rate_pct']}%)"
                )
        else:
            st.caption("No sample context available for hit-rate check.")


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
                "most_stable_avg",
                "stability_label",
                "overall_confidence",
                "profile",
                "vs_sample_size",
                "opponent_sample_adequate",
            ]
        ],
        width="stretch",
    )
