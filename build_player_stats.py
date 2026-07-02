import pandas as pd
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'ipl.csv')
OUT_PATH = os.path.join(CURRENT_DIR, 'player_stats.csv')

print("Loading ipl.csv...")
df = pd.read_csv(DATA_PATH)
df['year'] = pd.to_datetime(df['date'], dayfirst=False, errors='coerce').dt.year

fix = {'Rising Pune Supergiant': 'Rising Pune Supergiants', 'Royal Challengers Bengaluru': 'Royal Challengers Bangalore'}
df['bat_team'] = df['bat_team'].replace(fix)
df['bowl_team'] = df['bowl_team'].replace(fix)

df['wb'] = df.groupby(['mid', 'bat_team'])['wickets'].transform(lambda x: x.diff() > 0)
df['wb'] = df['wb'].fillna(False)

print("Computing batting stats...")
bat = df.groupby(['mid', 'batsman']).agg(runs=('striker', 'sum'), balls=('striker', 'count')).reset_index()
bt = df[['mid', 'batsman', 'bat_team']].drop_duplicates(['mid', 'batsman'])
bat = bat.merge(bt, on=['mid', 'batsman']).rename(columns={'bat_team': 'team'})
meta = df[['mid', 'year']].drop_duplicates('mid')
bat = bat.merge(meta, on='mid')

bat_season = bat.groupby(['batsman', 'year']).agg(
    team=('team', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ''),
    runs=('runs', 'sum'), balls=('balls', 'sum'), matches=('mid', 'nunique')
).reset_index()
bat_season['avg'] = (bat_season['runs'] / bat_season['matches'].clip(1)).round(1)
bat_season['sr'] = (bat_season['runs'] / bat_season['balls'].clip(1) * 100).round(1)

print("Computing bowling stats...")
bowl = df.groupby(['mid', 'bowler']).agg(
    wickets=('wb', 'sum'), conceded=('striker', 'sum'), balls=('striker', 'count')
).reset_index()
bowl['wickets'] = bowl['wickets'].astype(int)
bwt = df[['mid', 'bowler', 'bowl_team']].drop_duplicates(['mid', 'bowler'])
bowl = bowl.merge(bwt, on=['mid', 'bowler']).rename(columns={'bowl_team': 'team'})
bowl = bowl.merge(meta, on='mid')

bowl_season = bowl.groupby(['bowler', 'year']).agg(
    team=('team', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else ''),
    wickets=('wickets', 'sum'), conceded=('conceded', 'sum'),
    balls=('balls', 'sum'), matches=('mid', 'nunique')
).reset_index()
bowl_season['econ'] = (bowl_season['conceded'] / (bowl_season['balls'].clip(1) / 6)).round(2)

print("Computing roles...")
bat_match_count = bat.groupby(['batsman', 'year']).agg(bat_matches=('mid', 'nunique')).reset_index()
bowl_match_count = bowl.groupby(['bowler', 'year']).agg(bowl_matches=('mid', 'nunique')).reset_index()

all_players = pd.merge(
    bat_match_count, bowl_match_count,
    left_on=['batsman', 'year'], right_on=['bowler', 'year'], how='outer'
).fillna(0)
all_players['player'] = all_players['batsman'].fillna(all_players['bowler'])

def detect_role(row):
    bm = row['bat_matches']
    bwm = row['bowl_matches']
    total = max(bm, bwm)
    if total == 0:
        return 'bat'
    bp = bm / total
    wp = bwm / total
    if wp >= 0.4 and bp >= 0.25:
        return 'ar'
    if wp >= 0.5:
        return 'bowl'
    if bp >= 0.7 and wp >= 0.2:
        return 'ar'
    if bp >= 0.6:
        return 'bat'
    if bp >= 0.4 and wp >= 0.15:
        return 'ar'
    return 'bat' if bp > wp else 'bowl'

all_players['role'] = all_players.apply(detect_role, axis=1)

print("Merging stats...")
combined = pd.merge(
    bat_season, bowl_season,
    left_on=['batsman', 'year'], right_on=['bowler', 'year'], how='outer'
)
combined['player'] = combined['batsman'].fillna(combined['bowler'])
combined['team'] = combined['team_x'].fillna(combined['team_y'])
combined['runs'] = combined['runs'].fillna(0).astype(int)
combined['avg'] = combined['avg'].fillna(0)
combined['sr'] = combined['sr'].fillna(0)
combined['wickets'] = combined['wickets'].fillna(0).astype(int)
combined['econ'] = combined['econ'].fillna(0)
combined['bat_matches'] = combined['matches_x'].fillna(0).astype(int)
combined['bowl_matches'] = combined['matches_y'].fillna(0).astype(int)

role_map = all_players.set_index(['player', 'year'])['role'].to_dict()
combined['role'] = combined.apply(
    lambda r: role_map.get((r['player'], int(r['year'])), 'bat'), axis=1
)

cols = ['player', 'year', 'team', 'runs', 'avg', 'sr', 'wickets', 'econ', 'bat_matches', 'bowl_matches', 'role']
result = combined[cols].sort_values(['player', 'year']).reset_index(drop=True)
result.to_csv(OUT_PATH, index=False)
print(f"Saved {len(result):,} rows to {OUT_PATH}")
print(f"Unique players: {result['player'].nunique()}")
print(f"Years: {int(result['year'].min())}-{int(result['year'].max())}")
