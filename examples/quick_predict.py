"""
Example: Quick IPL prediction using the enhanced model.
Run this after training or downloading ipl_models_enhanced.pkl
"""
import pickle, numpy as np

with open('ipl_models_enhanced.pkl', 'rb') as f:
    d = pickle.load(f)

gb_model = d['gb_model']
feature_names = d['feature_names']
team_players = d['team_season_players']
orig_features = d['orig_features']
win_model = d['win_model']

base_cols = [c.rstrip('_') for c in orig_features]

def build_feature_array(bat, bowl, yr_bat, yr_bowl, overs, runs, wickets, runs_l5, wkts_l5):
    arr = np.zeros(len(feature_names))
    balls_bowled = int(overs)*6 + round((overs - int(overs))*10)
    run_rate = runs / overs if overs > 0 else 0
    wkts_left = 10 - wickets
    base_vals = [runs, wickets, overs, runs_l5, wkts_l5, run_rate, balls_bowled, wkts_left]
    for fname, val in zip(base_cols, base_vals):
        if fname in feature_names:
            arr[feature_names.index(fname)] = val
    bat_key = f'{bat}|{yr_bat}'
    bowl_key = f'{bowl}|{yr_bowl}'
    top5 = team_players.get(bat_key, {}).get('top_batsmen', [])
    top3 = team_players.get(bowl_key, {}).get('top_bowlers', [])
    for i in range(5):
        if i < len(top5):
            b = top5[i]
            for suff, val in [('runs_season', b['runs']), ('avg', b['avg']), ('sr', b['sr'])]:
                col = f'top_bat_{i+1}_{suff}'
                if col in feature_names: arr[feature_names.index(col)] = val
    for i in range(3):
        if i < len(top3):
            b = top3[i]
            for suff, val in [('wickets_season', b['wickets']), ('econ', b['econ'])]:
                col = f'top_bowl_{i+1}_{suff}'
                if col in feature_names: arr[feature_names.index(col)] = val
    if f'bat_team_{bat}' in feature_names: arr[feature_names.index(f'bat_team_{bat}')] = 1
    if f'bowl_team_{bowl}' in feature_names: arr[feature_names.index(f'bowl_team_{bowl}')] = 1
    return arr

examples = [
    ('Chennai Super Kings', 'Mumbai Indians', 2021, 2021, 10, 80, 2, 35, 1, "CSK vs MI, 2021, 10ov 80/2"),
    ('Mumbai Indians', 'Chennai Super Kings', 2020, 2020, 6, 45, 3, 25, 2, "MI vs CSK, 2020, 6ov 45/3"),
    ('Kolkata Knight Riders', 'Royal Challengers Bangalore', 2024, 2024, 8, 65, 1, 40, 0, "KKR vs RCB, 2024, 8ov 65/1"),
    ('Rajasthan Royals', 'Sunrisers Hyderabad', 2022, 2022, 15, 130, 5, 48, 2, "RR vs SRH, 2022, 15ov 130/5"),
]

for bat, bowl, yb, ybw, ov, r, w, rl5, wl5, desc in examples:
    arr = build_feature_array(bat, bowl, yb, ybw, ov, r, w, rl5, wl5)
    pred = gb_model.predict([arr])[0]
    crr = r / ov if ov > 0 else 0
    proj_crr = int(r + crr * (20 - ov))
    wkts_left = 10 - w
    if crr > 8 and wkts_left >= 5 and ov >= 5:
        strength = min(1.0, (crr / 8 - 1) * 3)
        weight = min(0.55, strength * 0.4 + (wkts_left / 10) * 0.15)
        pred = int(pred * (1 - weight) + proj_crr * weight)
    print(f"{desc:45s} => Predicted final: {int(pred):3d} runs  (CRR proj: {proj_crr})")

# Win probability example
print()
f_arr = np.array([[175, 8, 85, 7, 10.625, 10.0, -0.625, 38, 1]])
wp = win_model.predict_proba(f_arr)[0][1] * 100
print(f"Chase 175, 85/3 at 12ov => Win: {wp:.1f}%")

f_arr2 = np.array([[160, 10, 60, 8, 6.0, 10.0, 4.0, 45, 1]])
wp2 = win_model.predict_proba(f_arr2)[0][1] * 100
print(f"Chase 160, 60/2 at 10ov => Win: {wp2:.1f}%")
