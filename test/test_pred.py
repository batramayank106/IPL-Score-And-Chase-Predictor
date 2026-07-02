import pickle, numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)

# Load model
with open(os.path.join(PROJECT_DIR, 'ipl_models_enhanced.pkl'), 'rb') as f:
    data = pickle.load(f)

gb_model = data['gb_model']
feature_names = list(data['feature_names'])
team_players = data.get('team_season_players', {})
all_teams = data.get('all_teams', [])
venues = data.get('venues', [])
orig_features = data.get('orig_features', [])
num_bat = data.get('num_bat_slots', 11)
num_bowl = data.get('num_bowl_slots', 5)

def build_arr(bat, bowl, overs, runs, wickets, runs_l5, wkts_l5, yr_bat, yr_bowl, venue,
              xi_players=None, bowl_players=None, bat_stats=None, bowl_stats=None):
    arr = np.zeros(len(feature_names))
    bballs = int(overs)*6 + round((overs-int(overs))*10)
    rr = runs/overs if overs>0 else 0
    wl = 10-wickets
    vavg = data.get('venue_avg_scores',{}).get(venue, 165)
    base_vals = [runs, wickets, overs, runs_l5, wkts_l5, rr, bballs, wl, vavg]
    for fname, val in zip(orig_features, base_vals):
        if fname in feature_names: arr[feature_names.index(fname)] = val
    for fname in ['venue_avg_bat_team', 'is_home']:
        if fname in feature_names: arr[feature_names.index(fname)] = 0

    bp_list = team_players.get(f"{bat}|{yr_bat}", {}).get('all_players', [])
    for i in range(min(num_bat, len(xi_players or []))):
        pname = xi_players[i] if xi_players and i < len(xi_players) else ''
        pdata = next((p for p in bp_list if p['name']==pname), {})
        for feat, val in [(f'top_bat_{i+1}_runs_season',pdata.get('runs',0)),
                          (f'top_bat_{i+1}_avg',pdata.get('avg',0)),
                          (f'top_bat_{i+1}_sr',pdata.get('sr',0))]:
            if feat in feature_names: arr[feature_names.index(feat)] = val
        playing=1; mr=0; mb=0
        if bat_stats and i < len(bat_stats) and bat_stats[i]:
            b=bat_stats[i]; playing=b.get('playing',1); mr=b.get('match_runs',0); mb=b.get('match_balls',0)
        elif bat_stats is None and wickets < 10 and runs > 0:
            n_batted = min(wickets + 2, 7)
            total_balls_bowled = int(overs * 6)
            if i < n_batted:
                pos_weight = max(1, n_batted - i)
                total_w = sum(max(1, n_batted - j) for j in range(n_batted))
                mr = max(1, int(runs * pos_weight / total_w))
                mb = max(1, int(total_balls_bowled * pos_weight / total_w))
            else:
                playing = 0
        for feat, val in [(f'top_bat_{i+1}_playing',playing),(f'top_bat_{i+1}_match_runs',mr),(f'top_bat_{i+1}_match_balls',mb)]:
            if feat in feature_names: arr[feature_names.index(feat)] = val
        if i < 3:
            for feat in [f'top_bat_{i+1}_form_avg', f'top_bat_{i+1}_form_sr']:
                if feat in feature_names: arr[feature_names.index(feat)] = 0

    bp_list = team_players.get(f"{bowl}|{yr_bowl}", {}).get('all_players', [])
    for i in range(min(num_bowl, len(bowl_players or []))):
        pname = bowl_players[i] if bowl_players and i < len(bowl_players) else ''
        pdata = next((p for p in bp_list if p['name']==pname), {})
        for feat, val in [(f'top_bowl_{i+1}_wickets_season',pdata.get('wickets',0)),
                          (f'top_bowl_{i+1}_econ',pdata.get('econ',0))]:
            if feat in feature_names: arr[feature_names.index(feat)] = val
        playing=1; mw=0; mruns=0; mballs=0
        if bowl_stats and i < len(bowl_stats) and bowl_stats[i]:
            b=bowl_stats[i]; playing=b.get('playing',1); mw=b.get('match_wickets',0); mruns=b.get('match_runs',0); mballs=b.get('match_balls',0)
        elif bowl_stats is None and runs > 0 and overs > 0:
            total_balls_bowled = int(overs * 6)
            n_bowlers = min(num_bowl, len(bowl_players or []))
            econs = [next((p for p in bp_list if p['name']==bowl_players[j]), {}).get('econ', 9) or 9 for j in range(n_bowlers)]
            e_sum = sum(econs)
            if e_sum > 0 and n_bowlers > 0:
                mballs = max(1, total_balls_bowled // n_bowlers)
                mruns = max(1, int(runs * econs[i] / e_sum))
        for feat, val in [(f'top_bowl_{i+1}_playing',playing),(f'top_bowl_{i+1}_match_wickets',mw),
                          (f'top_bowl_{i+1}_match_runs',mruns),(f'top_bowl_{i+1}_match_balls',mballs)]:
            if feat in feature_names: arr[feature_names.index(feat)] = val
        if i < 2:
            for feat in [f'top_bowl_{i+1}_form_wkts', f'top_bowl_{i+1}_form_econ']:
                if feat in feature_names: arr[feature_names.index(feat)] = 0

    bc = f'bat_team_{bat}'
    if bc in feature_names: arr[feature_names.index(bc)] = 1
    boc = f'bowl_team_{bowl}'
    if boc in feature_names: arr[feature_names.index(boc)] = 1
    return arr

# Test: 130/2 at 10 overs
bat = "Sunrisers Hyderabad"
bowl = "Mumbai Indians"
yr = 2026
venue = "Wankhede Stadium"

# Get XI
def get_xi(team, year):
    return team_players.get(f"{team}|{year}", {}).get('xi_names', [])
def get_all_players(team, year):
    return team_players.get(f"{team}|{year}", {}).get('all_players', [])

xi = get_xi(bat, yr)
bowl_p = get_all_players(bowl, yr)
bowl_xi = get_xi(bowl, yr)
bowl_players = [n for n in bowl_xi if next((x for x in bowl_p if x['name']==n), {}).get('wickets',0) > 0][:5]
if not bowl_players:
    bowlers_sorted = sorted([p for p in bowl_p if p.get('wickets',0) > 0], key=lambda x: -x.get('wickets',0))
    bowl_players = [p['name'] for p in bowlers_sorted[:5]]

print(f"Batting XI ({bat} {yr}): {xi}")
print(f"Bowling ({bowl} {yr}): {bowl_players}")
print()

# Test with our fix
overs, runs, wickets = 10, 130, 2
arr = build_arr(bat, bowl, overs, runs, wickets, 50, 1, yr, yr, venue, xi, bowl_players, None, None)
pred = gb_model.predict([arr])[0]
crr = runs/overs if overs > 0 else 0
proj = int(runs + crr * (20-overs))
wkts_left = 10 - wickets
if crr > 8 and wkts_left >= 5 and overs >= 5:
    strength = min(1.0, (crr / 8 - 1) * 3)
    weight = min(0.55, strength * 0.4 + (wkts_left / 10) * 0.15)
    pred_adj = int(pred * (1 - weight) + proj * weight)
else:
    pred_adj = int(pred)
print(f"--- Test: {runs}/{wickets} in {overs} overs ---")
print(f"Predicted: {pred:.0f}  |  Adjusted: {pred_adj}  |  Raw range: {pred-8:.0f}-{pred+8:.0f}")
print(f"Current RR: {crr:.2f}  |  Projected @ CRR: {proj}")
print()

# Print feature breakdown to debug
print("Feature breakdown:")
label_map = {
    'runs': 'Total Runs', 'wickets': 'Wickets Lost', 'overs': 'Overs',
    'run_rate': 'Run Rate', 'wickets_left': 'Wickets Left',
    'venue_avg_bat_team': 'Venue x Team',
}
for i in range(7):
    for s, k in [('Match Runs', 'match_runs'), ('Match Balls', 'match_balls')]:
        feat = f'top_bat_{i+1}_{k}'
        if feat in feature_names:
            val = arr[feature_names.index(feat)]
            if val > 0:
                print(f"  Bat{i+1} {k}: {val:.0f}")

for i in range(5):
    for s, k in [('Match Runs', 'match_runs'), ('Match Balls', 'match_balls')]:
        feat = f'top_bowl_{i+1}_{k}'
        if feat in feature_names:
            val = arr[feature_names.index(feat)]
            if val > 0:
                print(f"  Bowl{i+1} {k}: {val:.0f}")

# Also test without our fix (old behavior)
print("\n--- Old behavior (no dist) ---")
arr_old = np.zeros(len(feature_names))
for fname, val in zip(orig_features, base_vals):
    if fname in feature_names: arr_old[feature_names.index(fname)] = val
bc = f'bat_team_{bat}'
if bc in feature_names: arr_old[feature_names.index(bc)] = 1
boc = f'bowl_team_{bowl}'
if boc in feature_names: arr_old[feature_names.index(boc)] = 1
pred_old = gb_model.predict([arr_old])[0]
print(f"Old prediction (no player features): {pred_old:.0f}")
