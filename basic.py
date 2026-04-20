from nba_api.stats.endpoints import leaguegamefinder, playergamelog
from nba_api.stats.static import players, teams
import pandas as pd

# === CONFIG ===
PLAYER_NAME = "Jayson Tatum"
TEAM_NAME = "Miami Heat"
SEASON = '2024-25'
NUM_GAMES = 10
STAT_TARGETS = ["PTS", "REB", "AST", "FG3M", "FG3A"]

# === HELPERS ===
def get_team_abbreviation(name):
    return next(team['abbreviation'] for team in teams.get_teams() if team['full_name'] == name)

def get_player_id(name):
    return players.find_players_by_full_name(name)[0]['id']

# === DATA FETCH ===
opponent_abbr = get_team_abbreviation(TEAM_NAME)
player_id = get_player_id(PLAYER_NAME)

# Get player logs (entire season)
log = playergamelog.PlayerGameLog(player_id=player_id, season=SEASON).get_data_frames()[0]

# Filter only games vs opponent
matchups = log[log['MATCHUP'].str.contains(opponent_abbr)]

# Take the last 10
recent = matchups.head(NUM_GAMES)

# Extract selected stats
summary = recent[["GAME_DATE"] + STAT_TARGETS].reset_index(drop=True)

# === OUTPUT ===
print(f"\n📊 LAST {len(summary)} GAMES FOR {PLAYER_NAME} VS {TEAM_NAME}:\n")
print(summary.to_string(index=False))