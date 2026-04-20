from nba_api.stats.endpoints import leaguegamefinder, playergamelog, commonteamroster
from nba_api.stats.static import players, teams
import pandas as pd
import numpy as np

# === CONFIG ===
TEAM_A_NAME = "Boston Celtics"
TEAM_B_NAME = "Miami Heat"
PLAYER_NAME = "Jayson Tatum"
STAT_TARGETS = ["PTS", "REB", "AST", "FG3M", "FG3A"]
NUM_MATCHUPS = 5
NUM_RECENT_GAMES = 10
SEASON = '2024-25'

# === HELPERS ===
def get_team_id(team_name):
    return next(team['id'] for team in teams.get_teams() if team['full_name'] == team_name)

def get_player_id(player_name):
    return players.find_players_by_full_name(player_name)[0]['id']

def get_team_roster(team_id, season):
    roster_df = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    return set(roster_df['PLAYER'])

def analyze_stat_trend(player_log, stat, weight_decay=0.9):
    values = player_log[stat].astype(float).tolist()[:NUM_RECENT_GAMES]
    weights = [weight_decay ** i for i in range(len(values))]
    weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    consistency = round(np.std(values), 3)
    return round(weighted_avg, 2), consistency, values

# === STEP 1: Get Matchup History ===
team_a_id = get_team_id(TEAM_A_NAME)
teamb_id = get_team_id(TEAM_B_NAME)

games_df = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_a_id).get_data_frames()[0]
matchups = games_df[games_df['MATCHUP'].str.contains(TEAM_B_NAME.split()[-1])]
matchups = matchups.sort_values("GAME_DATE", ascending=False).head(NUM_MATCHUPS)

# === STEP 2: Calculate Retention Score ===
roster_now = get_team_roster(team_a_id, SEASON)
retention_scores = []

for _, row in matchups.iterrows():
    past_roster = set(row['STARTERS'].split(', ')) if 'STARTERS' in row else set()
    if past_roster:
        shared = len(roster_now & past_roster)
        score = shared / len(past_roster)
        retention_scores.append(score)
    else:
        retention_scores.append(0)

avg_retention = np.mean(retention_scores) if retention_scores else 0

# === STEP 3 & 4: Player Stat Analysis ===
player_id = get_player_id(PLAYER_NAME)
games = playergamelog.PlayerGameLog(player_id=player_id, season=SEASON).get_data_frames()[0]

result = {
    "player": PLAYER_NAME,
    "team": TEAM_A_NAME,
    "opponent": TEAM_B_NAME,
    "avg_retention_score": round(avg_retention, 3),
}

for stat in STAT_TARGETS:
    avg, consistency, raw = analyze_stat_trend(games, stat)
    result[f"weighted_{stat.lower()}_avg"] = avg
    result[f"{stat.lower()}_consistency"] = consistency
    result[f"{stat.lower()}_raw_stats"] = raw

# === REPORT ===
print("\n=== BETTING ANALYSIS REPORT ===")
print(pd.Series(result))