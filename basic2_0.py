from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players, teams
import pandas as pd
import time

# === CONFIG ===
PLAYER_NAME = "Jayson Tatum"
TEAM_NAME = "Miami Heat"
NUM_GAMES = 10
STAT_TARGETS = ["PTS", "REB", "AST", "FG3M", "FG3A"]
SEASON_YEARS = ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21", "2019-20"]  # You can go further back

# === HELPERS ===
def get_team_abbreviation(name):
    return next(team['abbreviation'] for team in teams.get_teams() if team['full_name'] == name)

def get_player_id(name):
    return players.find_players_by_full_name(name)[0]['id']

# === SETUP ===
opponent_abbr = get_team_abbreviation(TEAM_NAME)
player_id = get_player_id(PLAYER_NAME)
vs_team_history = pd.DataFrame()

# === PULL MATCHUP GAMES ACROSS SEASONS ===
for season in SEASON_YEARS:
    log = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    vs_team = log[log['MATCHUP'].str.contains(opponent_abbr)]
    vs_team_history = pd.concat([vs_team_history, vs_team], ignore_index=True)

    if len(vs_team_history) >= NUM_GAMES:
        break

    time.sleep(0.6)  # avoid rate limits

vs_team_history = vs_team_history.head(NUM_GAMES).reset_index(drop=True)
vs_team_stats = vs_team_history[["SEASON_ID", "GAME_DATE", "MATCHUP"] + STAT_TARGETS]

# === GET OVERALL STATS FOR MOST RECENT SEASON ===
log_recent = playergamelog.PlayerGameLog(player_id=player_id, season=SEASON_YEARS[0]).get_data_frames()[0]
overall_stats = log_recent.head(NUM_GAMES)[["GAME_DATE", "MATCHUP"] + STAT_TARGETS]

# === DISPLAY ===
print(f"\n📊 LAST {len(vs_team_stats)} GAMES FOR {PLAYER_NAME} VS {TEAM_NAME} (MULTI-SEASON):\n")
print(vs_team_stats.to_string(index=False))

print(f"\n📈 LAST {len(overall_stats)} GAMES FOR {PLAYER_NAME} (OVERALL - {SEASON_YEARS[0]}):\n")
print(overall_stats.to_string(index=False))