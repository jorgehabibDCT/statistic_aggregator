import streamlit as st
from nba_api.stats.endpoints import playergamelog, commonteamroster
from nba_api.stats.static import players, teams
import pandas as pd
import time

from analysis.metrics import (
    add_combo_stats,
    apply_outlier_exclusion,
    compute_hit_rate,
    generate_benchmark_thresholds,
    run_comparison_engine,
)
from analysis.team_analysis import (
    build_player_summary_row,
    build_signal_rows,
    finalize_signal_board,
)
from analysis.profiles import infer_player_profile
from analysis.summaries import build_natural_language_summary
from ui.components import (
    format_stat_label,
    render_comparison_table,
    render_hit_rate_panel,
    render_low_sample_warning,
    render_summary_cards,
    render_summary_text,
)
from ui.charts import render_compact_comparison_charts
from ui.team_components import render_team_summary_table, render_top_signals_panel
from utils.constants import (
    BASE_STATS,
    COMBO_STATS,
    DEFAULT_OVERALL_SAMPLE,
    DEFAULT_VS_SAMPLE,
    EDGE_THRESHOLDS,
    LOW_SAMPLE_THRESHOLD,
    MIN_ANALYSIS_SAMPLE,
    SAMPLE_OPTIONS,
    STABILITY_THRESHOLDS,
)

st.set_page_config(page_title="NBA Player Matchup Stats", layout="centered")
st.title("Little Bucket Book")
st.caption("Matchup trend analysis for player role and performance splits.")

APP_VERSION = "v0.3.0"

# === SETUP ===
team_list = sorted(teams.get_teams(), key=lambda x: x['full_name'])
team_names = [team['full_name'] for team in team_list]
stat_targets = ["PTS", "REB", "AST", "FG3M", "FG3A", "STL", "BLK"]
seasons = ["2025-26","2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20"]

# === HELPERS ===
def get_team_by_name(name):
    return next(team for team in team_list if team['full_name'] == name)

def get_team_abbreviation(name):
    return get_team_by_name(name)['abbreviation']

def get_team_id(name):
    return get_team_by_name(name)['id']

def get_player_id(name):
    return players.find_players_by_full_name(name)[0]['id']


@st.cache_data(ttl=3600)
def fetch_roster(team_id, season):
    df = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    return df['PLAYER'].tolist()


@st.cache_data(ttl=3600)
def fetch_player_log(player_id, season):
    return playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]


@st.cache_data(ttl=3600)
def filter_games_vs_opponent(log_df, opp_abbr):
    return log_df[log_df["MATCHUP"].str.contains(opp_abbr, na=False)].copy()


@st.cache_data(ttl=900)
def build_cached_stat_results(vs_games, overall_games, analysis_stats):
    return run_comparison_engine(
        vs_games,
        overall_games,
        list(analysis_stats),
        EDGE_THRESHOLDS,
        STABILITY_THRESHOLDS,
    )


def _run_single_player_analysis(
    player_name_to_analyze,
    opponent_abbr,
    num_games_value,
    vs_sample_value,
    overall_sample_value,
    include_combo,
    exclude_high,
    exclude_low,
):
    player_id = get_player_id(player_name_to_analyze)
    all_games_local = pd.DataFrame()
    for season in seasons:
        log = fetch_player_log(player_id, season)
        vs_team = filter_games_vs_opponent(log, opponent_abbr)
        all_games_local = pd.concat([all_games_local, vs_team])
        if len(all_games_local) >= num_games_value:
            break
        time.sleep(0.6)

    all_games_local = all_games_local.head(num_games_value)
    if all_games_local.empty:
        return None

    log_recent = fetch_player_log(player_id, seasons[0])
    if log_recent.empty:
        return None

    vs_for_analysis_local = add_combo_stats(all_games_local.head(vs_sample_value).copy())
    overall_for_analysis_local = add_combo_stats(log_recent.head(overall_sample_value).copy())

    analysis_pool_stats_local = BASE_STATS + COMBO_STATS
    if exclude_high or exclude_low:
        vs_for_analysis_local = apply_outlier_exclusion(
            vs_for_analysis_local,
            analysis_pool_stats_local,
            exclude_high=exclude_high,
            exclude_low=exclude_low,
        )
        overall_for_analysis_local = apply_outlier_exclusion(
            overall_for_analysis_local,
            analysis_pool_stats_local,
            exclude_high=exclude_high,
            exclude_low=exclude_low,
        )

    if len(vs_for_analysis_local) < MIN_ANALYSIS_SAMPLE or len(overall_for_analysis_local) < MIN_ANALYSIS_SAMPLE:
        return None

    analysis_stats_local = BASE_STATS + (COMBO_STATS if include_combo else [])
    stat_results_local = build_cached_stat_results(
        vs_for_analysis_local,
        overall_for_analysis_local,
        tuple(sorted(analysis_stats_local)),
    )
    if stat_results_local.empty:
        return None

    profile_local = infer_player_profile(stat_results_local)
    return {
        "all_games": all_games_local,
        "log_recent": log_recent,
        "vs_for_analysis": vs_for_analysis_local,
        "overall_for_analysis": overall_for_analysis_local,
        "stat_results": stat_results_local,
        "profile": profile_local,
    }

# === INPUTS ===
selected_team = st.selectbox("Select Team", team_names, index=team_names.index("Boston Celtics"))
team_id = get_team_id(selected_team)
roster_players = fetch_roster(team_id, seasons[0])

if not roster_players:
    st.error("No roster data found for the selected team/season.")
    st.stop()

sorted_roster_players = sorted(roster_players)
if "player_select" not in st.session_state:
    st.session_state["player_select"] = "Jayson Tatum" if "Jayson Tatum" in sorted_roster_players else sorted_roster_players[0]
elif st.session_state["player_select"] not in sorted_roster_players:
    st.session_state["player_select"] = sorted_roster_players[0]

player_name = st.selectbox("Select Player", sorted_roster_players, key="player_select")
opponent_team = st.selectbox("Select Opponent Team", [t for t in team_names if t != selected_team], index=team_names.index("Miami Heat"))
num_games = st.slider("Number of Games", 1, 20, 10)

st.markdown("### Insight Controls")
use_same_sample = st.checkbox("Use same sample size for both splits", value=True)
if use_same_sample:
    shared_sample = st.selectbox("Sample Size", SAMPLE_OPTIONS, index=SAMPLE_OPTIONS.index(DEFAULT_VS_SAMPLE))
    vs_sample_size = shared_sample
    overall_sample_size = shared_sample
else:
    vs_sample_size = st.selectbox("Vs Opponent Sample Size", SAMPLE_OPTIONS, index=SAMPLE_OPTIONS.index(DEFAULT_VS_SAMPLE))
    overall_sample_size = st.selectbox("Overall Sample Size", SAMPLE_OPTIONS, index=SAMPLE_OPTIONS.index(DEFAULT_OVERALL_SAMPLE))

show_combo_stats = st.checkbox("Include combo stats (PR, PA, RA, PRA)", value=True)
with st.expander("Advanced Insight Controls", expanded=False):
    exclude_high_outlier = st.checkbox("Exclude highest value in insight sample", value=False)
    exclude_low_outlier = st.checkbox("Exclude lowest value in insight sample", value=False)


def _team_inputs_snapshot():
    return (
        selected_team,
        opponent_team,
        num_games,
        vs_sample_size,
        overall_sample_size,
        show_combo_stats,
        exclude_high_outlier,
        exclude_low_outlier,
    )


st.markdown("## Team Analysis + Top Signals")
min_signal_confidence = st.slider("Signal Filter: Min Confidence", 0, 100, 55, key="team_min_conf")
min_signal_delta = st.number_input("Signal Filter: Min Delta", min_value=0.0, value=1.0, step=0.5, key="team_min_delta")
min_signal_sample = st.slider("Signal Filter: Min Opponent Sample", 1, 10, 5, key="team_min_sample")
signal_include_combo = st.checkbox("Signal Board: Include combo stats", value=True, key="team_include_combo")

if st.session_state.get("team_analysis_inputs_snapshot") != _team_inputs_snapshot():
    st.session_state["team_analysis_ready"] = False

if st.button("Run Team Analysis"):
    st.session_state["team_analysis_ready"] = True
    st.session_state["team_analysis_inputs_snapshot"] = _team_inputs_snapshot()

if st.session_state.get("team_analysis_ready"):
    opp_abbr_for_team = get_team_abbreviation(opponent_team)
    team_summary_rows = []
    signal_rows = []

    with st.spinner("Running team signal scan..."):
        for roster_player in sorted_roster_players:
            player_analysis = _run_single_player_analysis(
                roster_player,
                opp_abbr_for_team,
                num_games,
                vs_sample_size,
                overall_sample_size,
                show_combo_stats,
                exclude_high_outlier,
                exclude_low_outlier,
            )
            if not player_analysis:
                continue

            summary_row = build_player_summary_row(
                roster_player,
                player_analysis["stat_results"],
                player_analysis["profile"],
                LOW_SAMPLE_THRESHOLD,
            )
            if summary_row:
                team_summary_rows.append(summary_row)

            signal_rows.extend(
                build_signal_rows(
                    roster_player,
                    player_analysis["stat_results"],
                    player_analysis["profile"],
                    LOW_SAMPLE_THRESHOLD,
                )
            )

    team_summary_df = pd.DataFrame(team_summary_rows)
    signal_df = finalize_signal_board(signal_rows)

    if not signal_include_combo and not signal_df.empty:
        signal_df = signal_df[signal_df["stat"].isin(BASE_STATS)]

    if not signal_df.empty:
        signal_df = signal_df[
            (signal_df["confidence"] >= min_signal_confidence)
            & (signal_df["delta"] >= min_signal_delta)
            & (signal_df["vs_sample_size"] >= min_signal_sample)
            & (signal_df["opponent_sample_adequate"])
        ].reset_index(drop=True)

    st.session_state["team_summary_df"] = team_summary_df
    st.session_state["team_signal_df"] = signal_df

if "team_signal_df" in st.session_state and "team_summary_df" in st.session_state:
    render_top_signals_panel(st.session_state["team_signal_df"])
    render_team_summary_table(st.session_state["team_summary_df"])

    team_players = st.session_state["team_summary_df"]["player"].tolist() if not st.session_state["team_summary_df"].empty else []
    if team_players:
        quick_player = st.selectbox("Quick Select Player from Team Analysis", sorted(team_players), key="team_quick_player")
        if st.button("Load Player into Detail View"):
            st.session_state["player_select"] = quick_player
            st.session_state["analysis_ready"] = False
            st.rerun()


def _analysis_inputs_snapshot():
    """Snapshot of inputs that require re-running analysis when changed."""
    return (
        selected_team,
        player_name,
        opponent_team,
        num_games,
        use_same_sample,
        vs_sample_size,
        overall_sample_size,
        show_combo_stats,
        exclude_high_outlier,
        exclude_low_outlier,
    )


# Invalidate stale analysis when user changes controls above
if st.session_state.get("analysis_inputs_snapshot") != _analysis_inputs_snapshot():
    st.session_state["analysis_ready"] = False

if st.button("Run Analysis"):
    st.session_state["analysis_ready"] = True
    st.session_state["analysis_inputs_snapshot"] = _analysis_inputs_snapshot()
    # Fresh run: show hit-rate results once without requiring form submit first
    st.session_state["_hit_rate_results_shown"] = False
    # New analysis: pick up benchmark default threshold for current stat (do not clobber in-form edits)
    st.session_state.pop("hit_rate_threshold_input", None)

# === MAIN LOGIC ===
# Persist results across reruns (e.g. hit-rate widget changes); button alone would drop the block next run.
if st.session_state.get("analysis_ready"):
    try:
        opp_abbr = get_team_abbreviation(opponent_team)
        player_analysis = _run_single_player_analysis(
            player_name,
            opp_abbr,
            num_games,
            vs_sample_size,
            overall_sample_size,
            show_combo_stats,
            exclude_high_outlier,
            exclude_low_outlier,
        )
        if not player_analysis:
            st.warning("No games found vs the selected opponent for the queried seasons.")
            st.stop()

        all_games = player_analysis["all_games"]
        log_recent = player_analysis["log_recent"]
        vs_for_analysis = player_analysis["vs_for_analysis"]
        overall_for_analysis = player_analysis["overall_for_analysis"]
        stat_results = player_analysis["stat_results"]
        profile = player_analysis["profile"]

        st.subheader(f"📊 {player_name}'s Last {len(all_games)} Games vs {opponent_team}")
        st.dataframe(all_games[["SEASON_ID", "GAME_DATE", "MATCHUP"] + stat_targets])

        # Also show overall last 10 games
        overall_stats = log_recent.head(num_games)[["GAME_DATE", "MATCHUP"] + stat_targets]

        st.subheader(f"📈 Last {num_games} Overall Games")
        st.dataframe(overall_stats)

        if len(all_games) < vs_sample_size:
            st.info(
                f"Only {len(all_games)} opponent games available; requested {vs_sample_size} for insight calculations."
            )

        analysis_stats = BASE_STATS + (COMBO_STATS if show_combo_stats else [])
        # Deterministic option order across reruns
        hit_rate_options = tuple(sorted(analysis_stats))
        best_row = stat_results.sort_values("delta", ascending=False).iloc[0]
        worst_row = stat_results.sort_values("delta", ascending=True).iloc[0]
        stable_row = stat_results.sort_values("volatility_score", ascending=True).iloc[0]
        volatile_row = stat_results.sort_values("volatility_score", ascending=False).iloc[0]
        overall_confidence = int(stat_results["confidence"].mean())

        render_low_sample_warning(len(vs_for_analysis), LOW_SAMPLE_THRESHOLD)
        render_summary_cards(best_row, worst_row, stable_row, volatile_row, profile, overall_confidence)
        render_compact_comparison_charts(stat_results, stats=("PTS", "REB", "AST", "STL", "BLK"))
        render_comparison_table(stat_results)
        summary_text = build_natural_language_summary(stat_results, profile, overall_confidence)
        render_summary_text(summary_text)

        st.markdown("### Hit-Rate Controls")
        # Stable keys + deterministic options; session_state survives reruns (see analysis_ready gate above).
        if "hit_rate_stat_select" not in st.session_state:
            st.session_state["hit_rate_stat_select"] = "PTS" if "PTS" in hit_rate_options else hit_rate_options[0]
        elif st.session_state["hit_rate_stat_select"] not in hit_rate_options:
            st.session_state["hit_rate_stat_select"] = "PTS" if "PTS" in hit_rate_options else hit_rate_options[0]
            st.session_state.pop("hit_rate_threshold_input", None)

        sel_for_defaults = st.session_state["hit_rate_stat_select"]
        auto_thresholds = generate_benchmark_thresholds(overall_for_analysis, sel_for_defaults)
        default_threshold = (
            auto_thresholds[1]
            if len(auto_thresholds) >= 2
            else (auto_thresholds[0] if auto_thresholds else 10.0)
        )
        if "hit_rate_threshold_input" not in st.session_state:
            st.session_state["hit_rate_threshold_input"] = float(default_threshold)

        with st.form("hit_rate_form", clear_on_submit=False):
            st.selectbox(
                "Hit-Rate Stat",
                hit_rate_options,
                format_func=format_stat_label,
                key="hit_rate_stat_select",
            )
            st.number_input(
                "Hit-Rate Threshold (>=)",
                min_value=0.0,
                step=0.5,
                format="%.1f",
                key="hit_rate_threshold_input",
            )
            apply_hit_rate = st.form_submit_button("Apply hit-rate analysis")

        hit_rate_stat = st.session_state["hit_rate_stat_select"]
        auto_thresholds = generate_benchmark_thresholds(overall_for_analysis, hit_rate_stat)
        hit_rate_threshold = float(st.session_state["hit_rate_threshold_input"])

        if apply_hit_rate or not st.session_state.get("_hit_rate_results_shown"):
            if hit_rate_threshold < 0:
                st.warning("Threshold must be non-negative.")
                st.stop()

            vs_hit_rate = compute_hit_rate(vs_for_analysis, hit_rate_stat, hit_rate_threshold)
            overall_hit_rate = compute_hit_rate(overall_for_analysis, hit_rate_stat, hit_rate_threshold)
            render_hit_rate_panel(hit_rate_stat, hit_rate_threshold, vs_hit_rate, overall_hit_rate, auto_thresholds)
            st.session_state["_hit_rate_results_shown"] = True

        st.caption(
            f"Status: vs sample={len(vs_for_analysis)}, overall sample={len(overall_for_analysis)} | "
            f"outlier exclusion={'on' if (exclude_high_outlier or exclude_low_outlier) else 'off'} | {APP_VERSION}"
        )

    except Exception as e:
        st.error(f"API/network failure or unexpected error. Please retry in a moment. Details: {e}")