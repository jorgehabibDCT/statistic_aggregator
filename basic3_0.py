import streamlit as st
from nba_api.stats.endpoints import playergamelog, commonteamroster
from nba_api.stats.static import players, teams
import pandas as pd
import time

from analysis.metrics import add_combo_stats, run_comparison_engine
from analysis.profiles import infer_player_profile
from analysis.summaries import build_natural_language_summary
from ui.components import (
    render_comparison_table,
    render_low_sample_warning,
    render_summary_cards,
    render_summary_text,
)
from utils.constants import (
    BASE_STATS,
    COMBO_STATS,
    DEFAULT_OVERALL_SAMPLE,
    DEFAULT_VS_SAMPLE,
    EDGE_THRESHOLDS,
    LOW_SAMPLE_THRESHOLD,
    SAMPLE_OPTIONS,
    STABILITY_THRESHOLDS,
)

st.set_page_config(page_title="NBA Player Matchup Stats", layout="centered")
st.title("Little Bucket Book")

# === SETUP ===
team_list = sorted(teams.get_teams(), key=lambda x: x['full_name'])
team_names = [team['full_name'] for team in team_list]
stat_targets = ["PTS", "REB", "AST", "FG3M", "FG3A"]
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

def get_roster(team_id, season):
    df = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    return df['PLAYER'].tolist()

# === INPUTS ===
selected_team = st.selectbox("Select Team", team_names, index=team_names.index("Boston Celtics"))
team_id = get_team_id(selected_team)
roster_players = get_roster(team_id, seasons[0])

player_name = st.selectbox("Select Player", sorted(roster_players), index=roster_players.index("Jayson Tatum") if "Jayson Tatum" in roster_players else 0)
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

# === MAIN LOGIC ===
if st.button("Run Analysis"):
    try:
        opp_abbr = get_team_abbreviation(opponent_team)
        player_id = get_player_id(player_name)

        all_games = pd.DataFrame()
        for season in seasons:
            log = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            vs_team = log[log['MATCHUP'].str.contains(opp_abbr)]
            all_games = pd.concat([all_games, vs_team])
            if len(all_games) >= num_games:
                break
            time.sleep(0.6)  # Respect rate limit

        all_games = all_games.head(num_games)
        st.subheader(f"📊 {player_name}'s Last {len(all_games)} Games vs {opponent_team}")
        st.dataframe(all_games[["SEASON_ID", "GAME_DATE", "MATCHUP"] + stat_targets])

        # Also show overall last 10 games
        log_recent = playergamelog.PlayerGameLog(player_id=player_id, season=seasons[0]).get_data_frames()[0]
        overall_stats = log_recent.head(num_games)[["GAME_DATE", "MATCHUP"] + stat_targets]

        st.subheader(f"📈 Last {num_games} Overall Games")
        st.dataframe(overall_stats)

        # --- Insight layer (keeps existing tables intact) ---
        vs_for_analysis = all_games.head(vs_sample_size).copy()
        overall_for_analysis = log_recent.head(overall_sample_size).copy()

        vs_for_analysis = add_combo_stats(vs_for_analysis)
        overall_for_analysis = add_combo_stats(overall_for_analysis)

        analysis_stats = BASE_STATS + (COMBO_STATS if show_combo_stats else [])
        stat_results = run_comparison_engine(
            vs_for_analysis,
            overall_for_analysis,
            analysis_stats,
            EDGE_THRESHOLDS,
            STABILITY_THRESHOLDS,
        )

        if not stat_results.empty:
            best_row = stat_results.sort_values("delta", ascending=False).iloc[0]
            worst_row = stat_results.sort_values("delta", ascending=True).iloc[0]
            stable_row = stat_results.sort_values("volatility_score", ascending=True).iloc[0]
            volatile_row = stat_results.sort_values("volatility_score", ascending=False).iloc[0]
            overall_confidence = int(stat_results["confidence"].mean())
            profile = infer_player_profile(stat_results)

            render_low_sample_warning(len(vs_for_analysis), LOW_SAMPLE_THRESHOLD)
            render_summary_cards(best_row, worst_row, stable_row, volatile_row, profile, overall_confidence)
            render_comparison_table(stat_results)
            summary_text = build_natural_language_summary(stat_results, profile, overall_confidence)
            render_summary_text(summary_text)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")