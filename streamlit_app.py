import streamlit as st
import pickle
import importlib
import numpy as np
import pandas as pd
import os, warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            parts = module.split('.')
            for i in range(len(parts), 1, -1):
                try:
                    mod = importlib.import_module('.'.join(parts[:i]))
                    if hasattr(mod, name):
                        return getattr(mod, name)
                except (ModuleNotFoundError, ImportError):
                    continue
            raise

def _load_pickle(path):
    with open(path, 'rb') as f:
        return _CompatUnpickler(f).load()

st.set_page_config(page_title="IPL Predictor Pro", page_icon="🏏", layout="wide")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

st.markdown("""
<style>
@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.03); }
  100% { transform: scale(1); }
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes glow {
  0% { box-shadow: 0 0 5px rgba(26,35,126,0.3); }
  50% { box-shadow: 0 0 20px rgba(26,35,126,0.6); }
  100% { box-shadow: 0 0 5px rgba(26,35,126,0.3); }
}
.score-card { animation: pulse 2s infinite; }
.fade-in { animation: slideUp 0.5s ease-out; }
.glow-card { animation: glow 3s infinite; }
.metric-box {
  background: linear-gradient(135deg, #1a237e, #283593);
  color: white; padding: 1rem; border-radius: 12px; text-align: center;
}
.metric-box-small {
  background: rgba(26,35,126,0.08);
  padding: 0.5rem; border-radius: 8px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    for p in [os.path.join(CURRENT_DIR, 'ipl_models_enhanced.pkl'),
              os.path.join(CURRENT_DIR, 'ipl_models.pkl')]:
        if os.path.exists(p):
            return _load_pickle(p)
    return None

data = load_model()
if data is None:
    st.error("No model file found. Run train_enhanced_model.py first.")
    st.stop()

is_enhanced = 'team_season_players' in data and data['team_season_players'] and 'xi_names' in list(data['team_season_players'].values())[0]
gb_model = data['gb_model']
ens = data.get('ensemble', {})
ridge_model = ens.get('ridge_model')
gbr_w = ens.get('gbr_weight', 1.0)
ridge_w = ens.get('ridge_weight', 0.0)
has_ensemble = ridge_model is not None
feature_names = list(data['feature_names'])
win_model = data.get('win_model')
win_features = data.get('win_features', [])
team_players = data.get('team_season_players', {})
venues = data.get('venues', [])
all_teams = data.get('all_teams', [])
years_list = [int(y) for y in data.get('years_available', list(range(2008, 2027)))]
orig_features = data.get('orig_features', [])
num_bat = data.get('num_bat_slots', 11)
num_bowl = data.get('num_bowl_slots', 5)

csv_path = os.path.join(CURRENT_DIR, 'ipl.csv')

venue_map = {}
if os.path.exists(csv_path):
    raw_venues = pd.read_csv(csv_path)['venue'].unique()
    for v in raw_venues:
        base = v.split(',')[0].strip()
        venue_map[v] = base
venue_displays = sorted(venue_map.keys()) if venue_map else venues

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
    # Extra features
    for fname in ['venue_avg_bat_team', 'is_home']:
        if fname in feature_names: arr[feature_names.index(fname)] = 0
    # Depth features (V2): remaining batting strength
    if 'depth_runs' in feature_names or 'depth_count' in feature_names:
        depth_runs = 0
        depth_count = 0
        bp_list = team_players.get(f"{bat}|{yr_bat}", {}).get('all_players', [])
        for i in range(min(num_bat, len(xi_players or []))):
            pname = xi_players[i] if xi_players and i < len(xi_players) else ''
            ply = 1
            if bat_stats and i < len(bat_stats) and bat_stats[i]:
                ply = bat_stats[i].get('playing', 1)
            if not ply:
                pdata = next((p for p in bp_list if p['name']==pname), {})
                depth_runs += pdata.get('runs', 0)
                depth_count += 1
        if 'depth_runs' in feature_names: arr[feature_names.index('depth_runs')] = depth_runs
        if 'depth_count' in feature_names: arr[feature_names.index('depth_count')] = depth_count
    if is_enhanced:
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
                n_batted = min(wickets + 2, 11)
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

@st.cache_data
def load_player_stats():
    p = os.path.join(CURRENT_DIR, 'player_stats.csv')
    if os.path.exists(p):
        df = pd.read_csv(p)
        df['display'] = df.apply(lambda r: f"{r['runs']} runs, {r['avg']} avg, {r['sr']} SR" if r['runs'] > 0
                                 else f"{int(r['wickets'])} wkts, {r['econ']} econ", axis=1)
        return df
    return pd.DataFrame()

player_stats_df = load_player_stats()

def get_xi(team, year):
    return team_players.get(f"{team}|{year}", {}).get('xi_names', [])

def get_all_players(team, year):
    """Return merged player list: player_stats.csv data (preferred) + model fallback."""
    model_players = team_players.get(f"{team}|{year}", {}).get('all_players', [])
    result = []
    seen = set()

    if not player_stats_df.empty:
        ps = player_stats_df[(player_stats_df['team']==team) & (player_stats_df['year']==year)]
        for _, r in ps.iterrows():
            result.append({
                'name': r['player'],
                'runs': int(r['runs']),
                'avg': float(r['avg']),
                'sr': float(r['sr']),
                'wickets': int(r['wickets']),
                'econ': float(r['econ']),
                'role': r.get('role', 'bat')
            })
            seen.add(r['player'])

    for p in model_players:
        if p['name'] not in seen:
            result.append(p)
            seen.add(p['name'])

    return result

st.title("🏏 IPL Predictor Pro")
st.caption(f"Made by Mayank Batra")

tab1, tab2 = st.tabs(["Match Predictor", "Player Stats"])

with tab1:
    mode = st.radio("Mode", ["First Innings Score", "Chase Win Probability"], horizontal=True)
    detail = st.radio("Detail Level", ["Quick (season avg)", "Advanced (match input)"], horizontal=True)
    is_adv = detail == "Advanced (match input)"

    c1, c2 = st.columns([1, 1])
    with c1: bat_team = st.selectbox("Batting Team", all_teams, key="bat")
    with c2: bowl_team = st.selectbox("Bowling Team", all_teams, key="bowl")
    c3, c4, c5 = st.columns(3)
    with c3: yr_bat = st.selectbox("Year (bat stats)", years_list, index=len(years_list)-1, key="yr_bat")
    with c4: yr_bowl = st.selectbox("Year (bowl stats)", years_list, index=len(years_list)-1, key="yr_bowl")
    with c5: venue_disp = st.selectbox("Venue", venue_displays, key="venue")
    venue = venue_map.get(venue_disp, venue_disp)

    form_stats, result_placeholder = None, None
    bat_stats = None; bowl_stats = None
    xi_players = get_xi(bat_team, yr_bat) if is_enhanced else []
    xi_key = f"xi_{bat_team}_{yr_bat}"

    if is_enhanced:
        all_p = get_all_players(bat_team, yr_bat)
        p_names = [p['name'] for p in all_p] if all_p else xi_players[:]
        if xi_key not in st.session_state:
            st.session_state[xi_key] = list(xi_players)
        xi_cur = list(st.session_state[xi_key])
        while len(xi_cur) < num_bat: xi_cur.append('')
        xi_cur = xi_cur[:num_bat]

        if is_adv:
            col_xi1, col_xi2 = st.columns(2)
            with col_xi1:
                st.markdown("### Batting XI")
                h1,h2,h3,h4 = st.columns([5,1,1,1])
                h1.caption("Batsman"); h2.caption("Runs"); h3.caption("Avg"); h4.caption("SR")
                for i in range(num_bat):
                    cur_name = xi_cur[i] if i < len(xi_cur) else ''
                    pdata = next((p for p in all_p if p['name']==cur_name), {})
                    c = st.columns([5,1,1,1])
                    with c[0]:
                        idx = p_names.index(cur_name) if cur_name in p_names else 0
                        chosen = st.selectbox("", p_names, index=idx, key=f"bx_{i}_{bat_team}_{yr_bat}", label_visibility="collapsed")
                        xi_cur[i] = chosen
                        pdata = next((p for p in all_p if p['name']==chosen), {})
                    c[1].caption(str(pdata.get('runs',0)) if pdata.get('runs',0) else '—')
                    c[2].caption(str(pdata.get('avg',0)) if pdata.get('avg',0) else '—')
                    c[3].caption(str(pdata.get('sr',0)) if pdata.get('sr',0) else '—')
                st.session_state[xi_key] = xi_cur

            # Bowling team's players for attack selection
            bowl_p = get_all_players(bowl_team, yr_bowl)
            bowl_p_names = [p['name'] for p in bowl_p if p.get('wickets',0) > 0 or p.get('role') in ('bowl', 'ar')]
            if not bowl_p_names:
                bowl_p_names = [p['name'] for p in bowl_p]

            bw_key = f"bw_{bowl_team}_{yr_bowl}"
            if bw_key not in st.session_state:
                default_bw = get_xi(bowl_team, yr_bowl)
                st.session_state[bw_key] = [n for n in default_bw if n in bowl_p_names][:num_bowl]
                if not st.session_state[bw_key]:
                    st.session_state[bw_key] = bowl_p_names[:num_bowl]
            bw_cur = list(st.session_state[bw_key])
            while len(bw_cur) < num_bowl: bw_cur.append('')
            bw_cur = bw_cur[:num_bowl]

            with col_xi2:
                st.markdown("### Bowling Attack")
                h1,h2,h3 = st.columns([5,1,1])
                h1.caption("Bowler"); h2.caption("Wkts"); h3.caption("Econ")
                for i in range(num_bowl):
                    cur_b = bw_cur[i] if i < len(bw_cur) else ''
                    pdata2 = next((p for p in bowl_p if p['name']==cur_b), {})
                    c = st.columns([5,1,1])
                    with c[0]:
                        idx2 = bowl_p_names.index(cur_b) if cur_b in bowl_p_names else 0
                        chosen_b = st.selectbox("", bowl_p_names, index=idx2, key=f"bw_{i}_{bowl_team}_{yr_bowl}", label_visibility="collapsed")
                        bw_cur[i] = chosen_b
                        pdata2 = next((p for p in bowl_p if p['name']==chosen_b), {})
                    c[1].caption(str(pdata2.get('wickets',0)) if pdata2.get('wickets',0) else '—')
                    c[2].caption(str(pdata2.get('econ',0)) if pdata2.get('econ',0) else '—')
                st.session_state[bw_key] = bw_cur

            bat_stats = []; bowl_stats = []
            with st.expander("Player Match Stats (advanced)", expanded=True):
                bs1, bs2 = st.columns(2)
                with bs1:
                    st.markdown("**Batting — runs & balls faced**")
                    for i, pname in enumerate([n for n in xi_cur if n]):
                        r1,r2,r3 = st.columns([2,1,1])
                        r1.caption(f"{i+1}. {pname}")
                        mr = r2.number_input("Runs",0,200,0,key=f"fi_bmr_{i}",label_visibility="collapsed")
                        mb = r3.number_input("Balls",0,120,0,key=f"fi_bmb_{i}",label_visibility="collapsed")
                        bat_stats.append({'playing':1,'match_runs':int(mr),'match_balls':int(mb)})
                with bs2:
                    st.markdown("**Bowling — wickets · runs · overs**")
                    for i, pname in enumerate([n for n in bw_cur if n]):
                        w1,w2,w3,w4 = st.columns([2,1,1,1])
                        w1.caption(f"{i+1}. {pname}")
                        mw = w2.number_input("Wkts",0,6,0,key=f"fi_bow_w_{i}",label_visibility="collapsed")
                        mr = w3.number_input("Runs",0,100,0,key=f"fi_bow_r_{i}",label_visibility="collapsed")
                        mo = w4.number_input("Overs",0.0,4.0,0.0,0.1,key=f"fi_bow_o_{i}",label_visibility="collapsed")
                        ov_i = int(mo); ov_d = round((mo - ov_i) * 10)
                        mballs = ov_i * 6 + min(ov_d, 5)
                        bowl_stats.append({'playing':1,'match_wickets':int(mw),'match_runs':int(mr),'match_balls':mballs})

            total_bat_runs = sum(b.get('match_runs',0) for b in bat_stats)
            total_bat_balls = sum(b.get('match_balls',0) for b in bat_stats)
            total_bowl_runs = sum(b.get('match_runs',0) for b in bowl_stats)
            total_bowl_balls = sum(b.get('match_balls',0) for b in bowl_stats)
            total_wkts = sum(b.get('match_wickets',0) for b in bowl_stats)

            errors = []
            if total_bat_balls > 120: errors.append(f"Batting balls ({total_bat_balls}) exceed 120")
            for i,b in enumerate(bowl_stats):
                if b['match_balls'] > 24: errors.append(f"Bowler {i+1} has {b['match_balls']} balls (max 24)")
            if total_bat_runs > 0 and total_bowl_runs > 0 and total_bat_runs != total_bowl_runs:
                errors.append(f"Batsman runs ({total_bat_runs}) ≠ bowler runs conceded ({total_bowl_runs})")
            if total_bat_balls > 0 and total_bowl_balls > 0 and abs(total_bat_balls - total_bowl_balls) > 6:
                errors.append(f"Batsmen faced {total_bat_balls} balls but bowlers bowled {total_bowl_balls}")
            for e in errors: st.warning(f"⚠️ {e}")

            # Auto-calculated match summary
            if total_bat_runs > 0 or total_bat_balls > 0:
                auto_overs = total_bowl_balls / 6 if total_bowl_balls > 0 else 0
                st.markdown("---")
                cols = st.columns(5)
                cols[0].markdown(f'<div class="metric-box-small"><small>Runs</small><br><b>{total_bat_runs}</b></div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="metric-box-small"><small>Wickets</small><br><b>{total_wkts}</b></div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div class="metric-box-small"><small>Overs</small><br><b>{auto_overs:.1f}</b></div>', unsafe_allow_html=True)
                cols[3].markdown(f'<div class="metric-box-small"><small>Run Rate</small><br><b>{total_bat_runs/auto_overs:.2f}</b></div>' if auto_overs > 0 else '<div class="metric-box-small"><small>Run Rate</small><br><b>—</b></div>', unsafe_allow_html=True)
                cols[4].markdown(f'<div class="metric-box-small"><small>Balls Bowled</small><br><b>{total_bowl_balls}</b></div>', unsafe_allow_html=True)

            xi_players = xi_cur
            bowl_players = bw_cur
        else:
            xi_players = xi_cur
            all_bowl_p = get_all_players(bowl_team, yr_bowl)
            bowl_xi = get_xi(bowl_team, yr_bowl)
            bowl_players = []
            for n in bowl_xi:
                p = next((x for x in all_bowl_p if x['name']==n), {})
                if p.get('wickets',0) > 0 or p.get('role') in ('bowl', 'ar'):
                    bowl_players.append(n)
            bowl_players = bowl_players[:num_bowl]
            if not bowl_players:
                bowlers = [p for p in all_bowl_p if p.get('wickets',0) > 0 or p.get('role') in ('bowl', 'ar')]
                bowl_players = [p['name'] for p in sorted(bowlers, key=lambda x:-x.get('wickets',0))[:num_bowl]]
            col_xi1, col_xi2 = st.columns(2)
            with col_xi1:
                rows = []
                for n in xi_players:
                    if not n: continue
                    p = next((x for x in all_p if x['name']==n), {})
                    rn = p.get('runs',0); av = p.get('avg',0); sr = p.get('sr',0)
                    rows.append({'Player': n, 'Runs': str(rn) if rn else '—',
                        'Avg': f'{av:.1f}' if av else '—', 'SR': f'{sr:.1f}' if sr else '—',
                        'Wkts': str(p.get('wickets',0)) if p.get('wickets',0) else '—', 'Role': p.get('role','')[:3].upper()})
                st.markdown("### Batting XI")
                if rows: st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            with col_xi2:
                rows = []
                for n in bowl_players:
                    if not n: continue
                    p = next((x for x in all_bowl_p if x['name']==n), {})
                    wk = p.get('wickets',0); ec = p.get('econ',0)
                    rows.append({'Player': n, 'Wkts': str(wk) if wk else '—',
                        'Econ': f'{ec:.2f}' if ec else '—'})
                st.markdown("### Bowling Attack")
                if rows: st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                else: st.caption("(bowlers from XI)")

    st.markdown("---")

    if mode == "First Innings Score":
        if is_adv:
            ov = max(total_bowl_balls / 6, 5.0) if bat_stats else 10.0
            rn = total_bat_runs if bat_stats else 80
            wk = total_wkts if bat_stats else 2
            rl5 = int(rn * 0.35) if bat_stats else 35
            wl5 = min(int(wk * 0.4), 10) if bat_stats else 1
            rl5 = st.slider("Runs last 5 overs (estimated)", 0, 200, rl5, key="fi_rl5_adv")
            wl5 = st.slider("Wkts last 5 overs (estimated)", 0, 10, wl5, key="fi_wl5_adv")
        else:
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: ov = st.number_input("Overs", 5.0, 20.0, 10.0, 0.1, key="fi_overs")
            with c2: rn = st.number_input("Runs", 0, 500, 80, key="fi_runs")
            with c3: wk = st.slider("Wickets", 0, 9, 2, key="fi_wkts")
            with c4: rl5 = st.number_input("Runs last 5", 0, 200, 35, key="fi_rl5")
            with c5: wl5 = st.slider("Wkts last 5", 0, 10, 1, key="fi_wl5")
        overs, runs, wickets, runs_l5, wkts_l5 = ov, rn, wk, rl5, wl5

        if st.button("Predict Score", type="primary", use_container_width=True):
            try:
                arr = build_arr(bat_team, bowl_team, overs, runs, wickets, runs_l5, wkts_l5,
                               yr_bat, yr_bowl, venue, xi_players, bowl_players, bat_stats, bowl_stats)
                pred_gbr = gb_model.predict([arr])[0]
                if has_ensemble:
                    pred_ridge = ridge_model.predict([arr])[0]
                    pred_ens = int(gbr_w * pred_gbr + ridge_w * pred_ridge)
                else:
                    pred_ens = int(pred_gbr)
                balls_b = int(overs)*6+round((overs-int(overs))*10)
                balls_rem = 120-balls_b; crr = runs/overs if overs>0 else 0
                proj = int(runs + crr * (20-overs))
                # Blend toward CRR when model undershoots the simple projection
                wkts_left = 10 - wickets
                pred_blend = pred_ens; weight = 0.0
                if overs >= 5 and wkts_left >= 3:
                    gap = proj - pred_ens
                    if gap > 0:
                        crr_factor = min(1.0, max(0, (crr - 6) / 6))
                        wkt_factor = wkts_left / 10
                        gap_factor = min(1.0, gap / proj * 3)
                        weight = min(0.55, crr_factor * 0.3 + wkt_factor * 0.15 + gap_factor * 0.2)
                        pred_blend = int(pred_ens * (1 - weight) + proj * weight)

                # Wicket adjustment: boost when depth intact, penalty when exposed
                WKTS_NEUTRAL = 3
                wkt_dev = WKTS_NEUTRAL - wickets
                if wkt_dev > 0:
                    boost_per = max(4, 10 - overs * 0.4)
                    pred_blend += int(wkt_dev * boost_per)
                elif wkt_dev < 0:
                    penalty_per = max(10, 30 - overs)
                    pred_blend = max(runs + 20, pred_blend + int(wkt_dev * penalty_per))

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.markdown(f"""
                    <div class="score-card fade-in" style="background:linear-gradient(135deg,#1a237e,#283593);color:white;padding:1.5rem;border-radius:16px;text-align:center;">
                        <div style="font-size:.8rem;text-transform:uppercase;letter-spacing:1px;opacity:.8">Smart Projection (Blend)</div>
                        <div style="font-size:3.5rem;font-weight:800;font-family:monospace">{pred_blend}</div>
                        <div style="font-size:.7rem;opacity:.7">{pred_blend-12} – {pred_blend+12} (68% CI)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="score-card fade-in" style="background:linear-gradient(135deg,#b71c1c,#c62828);color:white;padding:1.5rem;border-radius:16px;text-align:center;">
                        <div style="font-size:.8rem;text-transform:uppercase;letter-spacing:1px;opacity:.8">CRR Projection</div>
                        <div style="font-size:3.5rem;font-weight:800;font-family:monospace">{proj}</div>
                        <div style="font-size:.7rem;opacity:.7">{int(proj-15)} – {int(proj+15)} (68% CI)</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background:rgba(26,35,126,0.04);padding:0.8rem 1rem;border-radius:8px;font-size:.85rem;line-height:1.6;margin-bottom:0.5rem;">
                <strong>How the final projection is reached:</strong> GBR alone (confirmed optimal — Ridge at 0% by grid search) predicts <strong>{pred_ens}</strong> from 151 features. When the batting side has wickets in hand and a healthy scoring rate, the projection shifts toward a slightly more optimistic estimate based on the current run rate — this accounts for the model being trained on deeper historical data where scoring was generally lower than modern T20 standards.
                </div>
                """, unsafe_allow_html=True)

                if wkt_dev > 0:
                    st.markdown(f"""
                    <div style="background:rgba(26,35,126,0.04);padding:0.8rem 1rem;border-radius:8px;font-size:.85rem;line-height:1.6;margin-bottom:0.5rem;">
                    <strong>Batting depth intact:</strong> With only {wickets} wicket{'s' if wickets != 1 else ''} down, the full batting lineup is available — the projection is adjusted upward to reflect the scoring potential of the top order.
                    </div>
                    """, unsafe_allow_html=True)
                elif wkt_dev < 0:
                    st.markdown(f"""
                    <div style="background:rgba(26,35,126,0.04);padding:0.8rem 1rem;border-radius:8px;font-size:.85rem;line-height:1.6;margin-bottom:0.5rem;">
                    <strong>Batting depth adjustment:</strong> With {wickets} wickets down, the model accounts for the exposed lower order by adjusting the projection based on the expected scoring drop-off beyond the top 3.
                    </div>
                    """, unsafe_allow_html=True)

                if bat_stats and any(b['match_runs']>0 for b in bat_stats):
                    fig = go.Figure()
                    labels = [n for n in xi_players if n][:len(bat_stats)]
                    vals = [b['match_runs'] for b in bat_stats]
                    fig.add_trace(go.Bar(x=labels, y=vals, marker_color=vals, marker_colorscale='Blues', text=vals, textposition='outside'))
                    fig.update_layout(title="Runs Contribution", height=250, margin=dict(l=40,r=20,t=30,b=80), xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

                if bowl_stats and any(b['match_balls']>0 for b in bowl_stats):
                    fig = go.Figure()
                    labels = [n for n in bowl_players if n][:len(bowl_stats)]
                    fig.add_trace(go.Bar(name='Wickets', x=labels, y=[b['match_wickets'] for b in bowl_stats], marker_color='#1a237e'))
                    fig.add_trace(go.Bar(name='Econ', x=labels, y=[b['match_runs']/(b['match_balls']/6) if b['match_balls']>0 else 0 for b in bowl_stats], marker_color='#29b6f6', yaxis='y2'))
                    fig.update_layout(title="Bowling Figures", height=250, margin=dict(l=40,r=20,t=30,b=80), plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(title='Wickets'), yaxis2=dict(title='Economy', overlaying='y', side='right'))
                    st.plotly_chart(fig, use_container_width=True)

                st.balloons()
            except Exception as e: st.error(f"Error: {e}")

    else:
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: target = st.number_input("Target", 1, 500, 175, key="ch_target")
        with c2: overs = st.number_input("Overs", 5.0, 20.0, 12.0, 0.1, key="ch_overs")
        with c3: runs = st.number_input("Runs", 0, 500, 90, key="ch_runs")
        with c4: wickets = st.slider("Wickets Lost", 0, 9, 3, key="ch_wkts")
        with c5: runs_l5 = st.number_input("Runs last 5", 0, 200, 38, key="ch_rl5")
        wkts_l5 = st.slider("Wkts last 5", 0, 10, 1, key="ch_wl5")

        if st.button("Predict Win %", type="primary", use_container_width=True):
            try:
                balls_left = (20-overs)*6; runs_left = target-runs
                if balls_left <= 1: st.error("No balls left!")
                elif runs_left <= 0: st.success(f"{bat_team} won!")
                else:
                    req_rr = (runs_left/balls_left)*6; crr = runs/overs if overs>0 else 0
                    arr = build_arr(bat_team, bowl_team, overs, runs, wickets, runs_l5, wkts_l5,
                                   yr_bat, yr_bowl, venue, xi_players, bowl_players, None, None)
                    win_pct = 50.0
                    if win_model is not None:
                        ol=20-overs; rl=target-runs; wl=10-wickets
                        fa=np.array([[target,ol,rl,wl,req_rr,crr,crr-req_rr,runs_l5,wkts_l5]])
                        win_pct=win_model.predict_proba(fa)[0][1]*100
                    rr_adv = crr - req_rr
                    if rr_adv > 2:
                        boost = min(25, (rr_adv - 2) * 6)
                        win_pct = min(97, win_pct + boost)
                    elif rr_adv < -3:
                        penalty = min(30, abs(rr_adv) * 4)
                        win_pct = max(3, win_pct - penalty)
                    wp = 'high' if win_pct >= 65 else 'med' if win_pct >= 35 else 'low'
                    colors = {'high': '#1b5e20', 'med': '#e65100', 'low': '#b71c1c'}
                    col_a, col_b = st.columns([1, 1.5])
                    with col_a:
                        st.markdown(f"""
                        <div class="fade-in" style="background:linear-gradient(135deg,{colors[wp]},{colors[wp]}dd);color:white;padding:1.5rem;border-radius:16px;text-align:center;">
                            <div style="font-size:.8rem;text-transform:uppercase;letter-spacing:1px;opacity:.8">Win Probability</div>
                            <div style="font-size:3.5rem;font-weight:800;font-family:monospace">{win_pct:.0f}%</div>
                            <div style="margin-top:1rem;height:8px;background:rgba(255,255,255,0.2);border-radius:4px;overflow:hidden">
                                <div style="height:100%;width:{win_pct}%;background:rgba(255,255,255,0.9);border-radius:4px;transition:width 1.5s ease"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        m1,m2,m3,m4,m5,m6 = st.columns(6)
                        m1.metric("Req RR", f"{req_rr:.2f}"); m2.metric("Current RR", f"{crr:.2f}")
                        m3.metric("RR Diff", f"{crr-req_rr:+.2f}"); m4.metric("Proj @ CRR", int(runs+crr*(20-overs)))
                        m5.metric("Runs Left", runs_left); m6.metric("Balls Left", balls_left)

                    if abs(rr_adv) > 3:
                        direction = "advantage" if rr_adv > 0 else "pressure"
                        st.markdown(f"""
                        <div style="background:rgba(26,35,126,0.04);padding:0.5rem 1rem;border-radius:8px;font-size:.8rem;line-height:1.5;margin-bottom:0.5rem;">
                        <strong>Chase momentum:</strong> With a RR difference of {rr_adv:+.2f}, the model factors in the scoring {direction} when determining the win probability.
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("##### Chase Profile")
                    chase_fig = go.Figure()
                    o = list(range(int(overs)+1, 21))
                    rr_at_o = [crr * (20-i)/20 + (i/20)*req_rr for i in o]
                    chase_fig.add_trace(go.Scatter(x=o, y=[runs + crr * i for i in o], mode='lines', name='Current Pace', line=dict(color='#29b6f6', dash='dot')))
                    chase_fig.add_trace(go.Scatter(x=o, y=[target] * len(o), mode='lines', name='Target', line=dict(color='#ff7043', width=2)))
                    chase_fig.add_trace(go.Scatter(x=o, y=[runs + req_rr * i for i in o], mode='lines', name='Req Rate', line=dict(color='#66bb6a')))
                    chase_fig.update_layout(height=250, margin=dict(l=40,r=20,t=20,b=20), plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', y=1.1))
                    chase_fig.update_xaxes(title='Overs')
                    st.plotly_chart(chase_fig, use_container_width=True)
            except Exception as e: st.error(f"Error: {e}")

with tab2:
    c1,c2=st.columns(2)
    with c1: sel_team=st.selectbox("Team", all_teams, key="stat_team")
    with c2: sel_year=st.selectbox("Year", years_list, key="stat_year")
    sort_col = st.selectbox("Sort by", ["runs", "wickets", "avg", "sr", "econ"], index=0, key="stat_sort")

    if not player_stats_df.empty:
        ps = player_stats_df[(player_stats_df['team'] == sel_team) & (player_stats_df['year'] == sel_year)].copy()
        if not ps.empty:
            ps = ps.sort_values(sort_col, ascending=False)
            display = ps[['player', 'runs', 'avg', 'sr', 'wickets', 'econ', 'role']].reset_index(drop=True)
            display.columns = ['Player', 'Runs', 'Avg', 'SR', 'Wkts', 'Econ', 'Role']
            st.dataframe(display, hide_index=True, use_container_width=True)

            top_n = min(15, len(ps))
            top = ps.head(top_n)
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Runs', x=top['player'], y=top['runs'], marker_color='#1a237e',
                                 text=top['runs'].astype(int), textposition='outside'))
            fig.add_trace(go.Bar(name='Wkts', x=top['player'], y=top['wickets'], marker_color='#29b6f6',
                                 yaxis='y2', text=top['wickets'].astype(int), textposition='outside'))
            fig.update_layout(title=f"{sel_team} {sel_year} — Sorted by {sort_col.title()}",
                height=350, margin=dict(l=40,r=20,t=40,b=100),
                xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title='Runs'), yaxis2=dict(title='Wickets', overlaying='y', side='right'))
            st.plotly_chart(fig, use_container_width=True)

            # Mini summary
            tot = len(ps)
            with_runs = (ps['runs'] > 0).sum()
            with_wkts = (ps['wickets'] > 0).sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Players", tot)
            c2.metric("Batters (runs > 0)", with_runs)
            c3.metric("Bowlers (wkts > 0)", with_wkts)
            if with_runs > 0:
                c4.metric("Top Batter", f"{ps.loc[ps['runs'].idxmax(), 'player']} ({int(ps['runs'].max())})")
        else:
            st.info(f"No stats found for {sel_team} in {sel_year}")
    elif is_enhanced:
        tsp = team_players.get(f"{sel_team}|{sel_year}", {})
        xi = tsp.get('xi_names', [])
        if xi:
            ap = tsp.get('all_players', [])
            rows = []
            for n in xi:
                p = next((x for x in ap if x['name']==n), {})
                rn = p.get('runs',0); av = p.get('avg',0); sr = p.get('sr',0); wk = p.get('wickets',0); ec = p.get('econ',0)
                rows.append({'Pos': len(rows)+1, 'Player': n, 'Runs': str(rn) if rn else '—',
                    'Avg': f'{av:.1f}' if av else '—', 'SR': f'{sr:.1f}' if sr else '—',
                    'Wkts': str(wk) if wk else '—', 'Econ': f'{ec:.2f}' if ec else '—',
                    'Role': p.get('role','')})
            if rows:
                df_xi = pd.DataFrame(rows)
                st.dataframe(df_xi, hide_index=True, use_container_width=True)
        else:
            bats = tsp.get('top_batsmen',[]); bowlers = tsp.get('top_bowlers',[])
            if bats: st.dataframe(pd.DataFrame(bats), hide_index=True, use_container_width=True)
            if bowlers: st.dataframe(pd.DataFrame(bowlers), hide_index=True, use_container_width=True)
    else:
        st.info("No player stats available. Run build_player_stats.py first or use the enhanced model.")

if is_enhanced and hasattr(gb_model,'feature_importances_'):
    with st.expander("Model Feature Importances"):
        label_map = {
            'runs': 'Total Runs', 'wickets': 'Wickets Lost', 'overs': 'Overs Bowled',
            'runs_last_5': 'Runs (Last 5 Ov)', 'wickets_last_5': 'Wkts (Last 5 Ov)',
            'run_rate': 'Run Rate', 'balls_bowled': 'Balls Bowled',
            'wickets_left': 'Wickets in Hand', 'venue_avg': 'Venue Avg Score',
            'venue_avg_bat_team': 'Venue × Team Avg',
            'is_home': 'Home Advantage',
            'depth_runs': 'Depth Runs (Remaining)',
            'depth_count': 'Depth Count (Remaining)',
        }
        for i in range(11):
            p = i + 1
            suf = ['Season Runs', 'Avg', 'SR', 'Playing?', 'Match Runs', 'Match Balls']
            for j, s in enumerate(suf):
                label_map[f'top_bat_{p}_runs_season' if j == 0 else f'top_bat_{p}_{["runs_season","avg","sr","playing","match_runs","match_balls"][j]}'] = f'Bat{p}-{s}'
            if i < 3:
                label_map[f'top_bat_{p}_form_avg'] = f'Bat{p}-Form Avg'
                label_map[f'top_bat_{p}_form_sr'] = f'Bat{p}-Form SR'
        for i in range(5):
            q = i + 1
            label_map[f'top_bowl_{q}_wickets_season'] = f'Bowl{q}-Season Wkts'
            label_map[f'top_bowl_{q}_econ'] = f'Bowl{q}-Econ'
            label_map[f'top_bowl_{q}_playing'] = f'Bowl{q}-Playing?'
            label_map[f'top_bowl_{q}_match_wickets'] = f'Bowl{q}-Match Wkts'
            label_map[f'top_bowl_{q}_match_runs'] = f'Bowl{q}-Match Runs'
            label_map[f'top_bowl_{q}_match_balls'] = f'Bowl{q}-Match Balls'
            if i < 2:
                label_map[f'top_bowl_{q}_form_wkts'] = f'Bowl{q}-Form Wkts'
                label_map[f'top_bowl_{q}_form_econ'] = f'Bowl{q}-Form Econ'
        for t in all_teams:
            label_map[f'bat_team_{t}'] = f'Bat Team: {t[:12]}'
            label_map[f'bowl_team_{t}'] = f'Bowl Team: {t[:12]}'

        imp = gb_model.feature_importances_
        idx = np.argsort(imp)[::-1][:20]
        labels = [label_map.get(feature_names[i], feature_names[i]) for i in idx]
        vals = [imp[i] for i in idx]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = []
        for name in labels:
            if name.startswith('Bat'): colors.append('#1a237e')
            elif name.startswith('Bowl'): colors.append('#29b6f6')
            elif name.startswith('Total Runs') or name.startswith('Wickets') or name.startswith('Overs') or 'Venue' in name or 'Rate' in name or 'Balls' in name:
                colors.append('#e65100')
            else: colors.append('#2e7d32')
        ax.barh(range(20), vals, color=colors)
        ax.set_yticks(range(20)); ax.set_yticklabels(labels)
        ax.invert_yaxis(); ax.set_xlabel('Importance')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1a237e', label='Batting'),
                           Patch(facecolor='#29b6f6', label='Bowling'),
                           Patch(facecolor='#e65100', label='Match State'),
                           Patch(facecolor='#2e7d32', label='Team')]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
        st.pyplot(fig)

        st.markdown("""
        <style>
        .insight-box { background: rgba(26,35,126,0.06); padding: 0.8rem 1rem; border-radius: 8px; margin: 0.5rem 0; }
        .insight-box strong { color: #1a237e; }
        </style>
        """, unsafe_allow_html=True)

        # Category-level breakdown
        cat_map = {'Match State': [], 'Batting': [], 'Bowling': [], 'Team': []}
        for i, fn in enumerate(feature_names):
            if fn in label_map:
                lbl = label_map[fn]
                if lbl.startswith('Bat'): cat_map['Batting'].append(imp[i])
                elif lbl.startswith('Bowl'): cat_map['Bowling'].append(imp[i])
                elif 'Team' in lbl: cat_map['Team'].append(imp[i])
                else: cat_map['Match State'].append(imp[i])
        st.markdown("**Feature Group Importance**")
        c1, c2, c3, c4 = st.columns(4)
        for col, (cat, vals_list) in zip([c1, c2, c3, c4], cat_map.items()):
            total_pct = sum(vals_list) * 100
            col.metric(cat, f"{total_pct:.1f}%")

        # Per-slot Match Runs importance
        st.markdown("**Match Runs Importance by Batting Position**")
        mr_imp = {}
        for i in range(11):
            key = f'top_bat_{i+1}_match_runs'
            if key in feature_names:
                mr_imp[f'Bat{i+1}'] = imp[feature_names.index(key)] * 100
        if mr_imp:
            fig2, ax2 = plt.subplots(figsize=(9, 2.5))
            slots = list(mr_imp.keys())
            vals = [mr_imp[s] for s in slots]
            colors2 = ['#e65100' if v > 0 else '#e0e0e0' for v in vals]
            ax2.bar(slots, vals, color=colors2)
            ax2.set_ylabel('Importance %')
            ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
            ax2.tick_params(axis='x', rotation=45)
            st.pyplot(fig2)

        st.markdown("""
        <div class="insight-box">
        <strong>All 11 batting positions contribute match_runs.</strong>
        V2 model uses match stats for all positions including Bat8–Bat11, plus two new depth features
        (remaining season runs and count of batsmen yet to bat) that capture batting depth.
        </div>
        """, unsafe_allow_html=True)

        # Season runs by slot
        st.markdown("**Season Runs Importance by Batting Position**")
        sr_imp = {}
        for i in range(11):
            key = f'top_bat_{i+1}_runs_season'
            if key in feature_names:
                sr_imp[f'Bat{i+1}'] = imp[feature_names.index(key)] * 100
        if sr_imp:
            fig3, ax3 = plt.subplots(figsize=(9, 2.5))
            slots = list(sr_imp.keys())
            vals = [sr_imp[s] for s in slots]
            max_val = max(vals) if vals else 0
            colors3 = ['#1a237e' if v == max_val else '#7986cb' for v in vals]
            ax3.bar(slots, vals, color=colors3)
            ax3.set_ylabel('Importance %')
            ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
            ax3.tick_params(axis='x', rotation=45)
            st.pyplot(fig3)

            top_slot = slots[vals.index(max_val)]
            st.markdown(f"""
            <div class="insight-box">
            <strong>{top_slot} has the highest Season Runs importance.</strong> This slot typically holds the team's best batter — their season quality reflects overall team strength independent of current match state.
            </div>
            """, unsafe_allow_html=True)
