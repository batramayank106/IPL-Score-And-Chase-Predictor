"""
IPL v8 — 11-player XI features, roles, memory-optimized
"""
import pandas as pd
import numpy as np
import pickle, warnings, os, time

np.random.seed(42)

warnings.filterwarnings('ignore')
np.random.seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'ipl.csv')
MODEL_OUT = os.path.join(CURRENT_DIR, 'ipl_models_enhanced.pkl')

print("LOADING", flush=True)
df = pd.read_csv(DATA_PATH)
KEEP = ['Chennai Super Kings','Deccan Chargers','Delhi Capitals','Delhi Daredevils',
    'Gujarat Lions','Gujarat Titans','Kings XI Punjab','Kochi Tuskers Kerala',
    'Kolkata Knight Riders','Lucknow Super Giants','Mumbai Indians','Pune Warriors',
    'Punjab Kings','Rajasthan Royals','Rising Pune Supergiants','Rising Pune Supergiant',
    'Royal Challengers Bangalore','Royal Challengers Bengaluru','Sunrisers Hyderabad']
df = df[df['bat_team'].isin(KEEP) & df['bowl_team'].isin(KEEP)].copy()
df['year'] = pd.to_datetime(df['date']).dt.year
fix = {'Rising Pune Supergiant':'Rising Pune Supergiants','Royal Challengers Bengaluru':'Royal Challengers Bangalore'}
df['bat_team']=df['bat_team'].replace(fix); df['bowl_team']=df['bowl_team'].replace(fix)
df['striker']=df['striker'].fillna(0)
df['venue_clean']=df['venue'].apply(lambda v: v.split(',')[0].strip())
df['venue_city']=df['venue'].apply(lambda v: v.split(',')[-1].strip() if ',' in v else '')

team_home_city={
    'Chennai Super Kings':'Chennai','Deccan Chargers':'Hyderabad',
    'Delhi Capitals':'Delhi','Delhi Daredevils':'Delhi',
    'Gujarat Lions':'Rajkot','Gujarat Titans':'Ahmedabad',
    'Kings XI Punjab':'Mohali','Kochi Tuskers Kerala':'Kochi',
    'Kolkata Knight Riders':'Kolkata','Lucknow Super Giants':'Lucknow',
    'Mumbai Indians':'Mumbai','Pune Warriors':'Pune',
    'Punjab Kings':'Mohali','Rajasthan Royals':'Jaipur',
    'Rising Pune Supergiants':'Pune','Rising Pune Supergiant':'Pune',
    'Royal Challengers Bangalore':'Bengaluru','Royal Challengers Bengaluru':'Bengaluru',
    'Sunrisers Hyderabad':'Hyderabad',
}
# Normalize venue_city: map alternate names
def norm_city(c):
    c=c.lower()
    if c in ('uppal','gachibowli'): return 'Hyderabad'
    if c in ('chepauk','chennai'): return 'Chennai'
    if 'bengaluru' in c or 'bangalore' in c: return 'Bengaluru'
    if 'mohali' in c or 'chandigarh' in c or 'mullanpur' in c: return 'Mohali'
    if 'kolkata' in c: return 'Kolkata'
    if 'mumbai' in c: return 'Mumbai'
    if 'delhi' in c: return 'Delhi'
    if 'jaipur' in c: return 'Jaipur'
    if 'lucknow' in c or 'kanpur' in c: return 'Lucknow'
    if 'ahmedabad' in c or 'motera' in c: return 'Ahmedabad'
    if 'pune' in c: return 'Pune'
    if 'rajkot' in c: return 'Rajkot'
    if 'kochi' in c: return 'Kochi'
    if 'dharamsala' in c or 'dharmashala' in c: return 'Dharamsala'
    if 'visakhapatnam' in c or 'vizag' in c: return 'Visakhapatnam'
    if 'guwahati' in c or 'barsapara' in c: return 'Guwahati'
    return c.title()
df['venue_city']=df['venue_city'].apply(norm_city)
df['is_home']=df.apply(lambda r: 1 if team_home_city.get(r['bat_team'],'').lower()==r['venue_city'].lower() else 0, axis=1)
print(f"  {len(df):,} rows, {df['mid'].nunique()} matches", flush=True)

mw = {}
for mid,g in df.groupby('mid'):
    t=list(g['bat_team'].unique())
    if len(t)<2: continue
    r1=g[g['bat_team']==t[0]]['runs'].max(); r2=g[g['bat_team']==t[1]]['runs'].max()
    if r1>r2: mw[mid]=t[0]
    elif r2>r1: mw[mid]=t[1]

va = df[df['overs']>=18].groupby('venue_clean')['total'].mean().to_dict()
va_bt = df[df['overs']>=18].groupby(['bat_team','venue_clean'])['total'].mean().to_dict()

df['wb'] = df.groupby(['mid','bat_team'])['wickets'].transform(lambda x: x.diff() > 0)
df['wb'] = df['wb'].fillna(False)

bs=df.groupby(['mid','batsman']).agg(runs=('striker','sum'),balls=('striker','count')).reset_index()
bt=df[['mid','batsman','bat_team']].drop_duplicates(['mid','batsman'])
bs=bs.merge(bt,on=['mid','batsman']).rename(columns={'bat_team':'team'})
bw=df.groupby(['mid','bowler']).agg(wkts=('wb','sum'),conc=('striker','sum'),balls=('striker','count')).reset_index()
bwt=df[['mid','bowler','bowl_team']].drop_duplicates(['mid','bowler'])
bw=bw.merge(bwt,on=['mid','bowler']).rename(columns={'bowl_team':'team'})
bw['wkts']=bw['wkts'].astype(int)
meta=df[['mid','year','date']].drop_duplicates('mid')
bs=bs.merge(meta,on='mid'); bw=bw.merge(meta,on='mid')
bs['date']=pd.to_datetime(bs['date']); bw['date']=pd.to_datetime(bw['date'])
print(f"  Batsman-match: {len(bs)}, Bowler-match: {len(bw)}", flush=True)

psb=bs.groupby(['batsman','year']).agg(tr=('runs','sum'),tb=('balls','sum'),m=('mid','nunique')).reset_index()
pst=bs.groupby(['batsman','year']).agg(team=('team',lambda x:x.mode().iloc[0])).reset_index()
psb=psb.merge(pst,on=['batsman','year'])
psb['avg']=(psb['tr']/psb['m'].clip(1)).round(1)
psb['sr']=(psb['tr']/psb['tb'].clip(1)*100).round(1)

psw=bw.groupby(['bowler','year']).agg(tw=('wkts','sum'),tc=('conc','sum'),tba=('balls','sum'),m=('mid','nunique')).reset_index()
pswt=bw.groupby(['bowler','year']).agg(team=('team',lambda x:x.mode().iloc[0])).reset_index()
psw=psw.merge(pswt,on=['bowler','year'])
psw['econ']=(psw['tc']/(psw['tba'].clip(1)/6)).round(2)

pos_rec=[]
for (mid,team),g in df.sort_index().groupby(['mid','bat_team']):
    seen=[]; [seen.append(r['batsman']) for _,r in g.iterrows() if r['batsman'] not in seen]
    for pos,p in enumerate(seen,1): pos_rec.append({'mid':mid,'team':team,'batsman':p,'pos':pos})
pos_df=pd.DataFrame(pos_rec).merge(meta[['mid','year']],on='mid')
bat_pos=pos_df.groupby(['batsman','year']).agg(pos=('pos',lambda x:int(x.mode().iloc[0]) if len(x.mode())>0 else 7)).reset_index()
psb=psb.merge(bat_pos,on=['batsman','year'],how='left'); psb['pos']=psb['pos'].fillna(7).astype(int)

def detect_role(bat_matches, bowl_matches, _unused=0):
    total = max(bat_matches, bowl_matches)
    if total == 0: return 'bat'
    bp = bat_matches / total
    wp = bowl_matches / total
    if wp >= 0.4 and bp >= 0.25: return 'ar'
    if wp >= 0.5: return 'bowl'
    if bp >= 0.7 and wp >= 0.2: return 'ar'
    if bp >= 0.6: return 'bat'
    if bp >= 0.4 and wp >= 0.15: return 'ar'
    return 'bat' if bp > wp else 'bowl'

print("BUILD ROSTER", flush=True)
full_roster = {}
for (team,year),g in psb.groupby(['team','year']):
    key = f"{team}|{int(year)}"
    roster = {'batsmen':[], 'bowlers':[], 'all_players':{}}
    for _,r in g.iterrows():
        p=r['batsman']
        bd=bw[(bw['team']==team)&(bw['year']==year)&(bw['bowler']==p)]
        role=detect_role(r['m'], bd['mid'].nunique(), r['m'])
        e={'name':p,'runs':int(r['tr']),'avg':r['avg'],'sr':r['sr'],
           'position':r['pos'],'matches':int(r['m']),'role':role,
           'wickets':int(bd['wkts'].sum()) if len(bd)>0 else 0,
           'econ':round(bd['conc'].sum()/(bd['balls'].sum()/6),2) if len(bd)>0 and bd['balls'].sum()>0 else 0}
        roster['batsmen'].append(e); roster['all_players'][p]=e
    for _,r in psw[(psw['team']==team)&(psw['year']==year)].iterrows():
        p=r['bowler']
        bd=bs[(bs['team']==team)&(bs['year']==year)&(bs['batsman']==p)]
        role=detect_role(bd['mid'].nunique(), r['m'], max(bd['mid'].nunique(),r['m']))
        e={'name':p,'wickets':int(r['tw']),'econ':r['econ'],'matches':int(r['m']),'role':role,
           'runs':int(bd['runs'].sum()) if len(bd)>0 else 0,
           'avg':round(bd['runs'].sum()/bd['mid'].nunique(),1) if len(bd)>0 and bd['mid'].nunique()>0 else 0}
        roster['bowlers'].append(e)
        if p not in roster['all_players']: roster['all_players'][p]=e
    full_roster[key]=roster

def build_xi(roster):
    bats=sorted(roster['batsmen'], key=lambda x:(-x['matches'],x['position']))
    bats_pure=[p for p in bats if p['role']=='bat']
    ars=[p for p in bats if p['role']=='ar']
    bowlers_all=sorted(roster['bowlers'], key=lambda x:-x['matches'])
    bowlers_pure=[p for p in bowlers_all if p['role']=='bowl' and p['name'] not in {x['name'] for x in bats}]
    xi_names=[]
    for p in bats_pure[:4]:
        if p['name'] not in xi_names: xi_names.append(p['name'])
    for p in ars[:2]:
        if p['name'] not in xi_names: xi_names.append(p['name'])
    for p in bowlers_pure:
        if len(xi_names)>=11: break
        if p['name'] not in xi_names: xi_names.append(p['name'])
    for p in ars[2:]+bats_pure[4:]:
        if len(xi_names)>=11: break
        if p['name'] not in xi_names: xi_names.append(p['name'])
    for p in bowlers_all:
        if len(xi_names)>=11: break
        if p['name'] not in xi_names: xi_names.append(p['name'])
    name_pos={}
    for p in roster['batsmen']:
        if p['name'] in xi_names: name_pos[p['name']]=p['position']
    for p in roster['bowlers']:
        if p['name'] in xi_names and p['name'] not in name_pos: name_pos[p['name']]=11
    return sorted(xi_names[:11], key=lambda n:name_pos.get(n,11))

for key,roster in full_roster.items():
    roster['xi_names'] = build_xi(roster)

# XI lookup for training: use XI names, get batting + bowling stats
NUM_BAT=11  # All 11 players in XI, each with batting stats
NUM_BOWL=5  # Top 5 bowlers from XI, each with bowling stats
tb_lookup={}
tw_lookup={}
for key,roster in full_roster.items():
    parts=key.split('|'); team=parts[0]; year=int(parts[1])
    xi_names=roster.get('xi_names',[])
    if not xi_names:
        xi_names=[p['name'] for p in sorted(roster['batsmen'],key=lambda x:-x['runs'])[:11]]
    
    # Batting stats for XI (all 11)
    name_map={p['name']:p for p in roster['batsmen']}
    fallback={p['name']:p for p in roster['bowlers']}
    bat_list=[]
    for n in xi_names:
        if n in name_map:
            p=name_map[n]
            bat_list.append({'name':n,'runs':p['runs'],'avg':p['avg'],'sr':p['sr'],'wickets':p.get('wickets',0),'econ':p.get('econ',0)})
        elif n in fallback:
            p=fallback[n]
            bat_list.append({'name':n,'runs':p.get('runs',0),'avg':p.get('avg',0),'sr':0,'wickets':p['wickets'],'econ':p['econ']})
    while len(bat_list)<NUM_BAT:
        bat_list.append({'name':'','runs':0,'avg':0,'sr':0,'wickets':0,'econ':0})
    tb_lookup[(team,year)]=bat_list[:NUM_BAT]
    
    # Bowling stats: pick bowlers/ARs from XI
    bowlers_in_xi = []
    added=set()
    # Find bowlers within XI
    for p in sorted(roster['bowlers'], key=lambda x:-x['wickets']):
        if p['name'] in xi_names and p['name'] not in added:
            bowlers_in_xi.append(p); added.add(p['name'])
    # Add ARs from XI who have bowling data
    if len(bowlers_in_xi)<NUM_BOWL:
        for p in roster['batsmen']:
            if p['name'] in xi_names and p['name'] not in added and p.get('wickets',0)>0:
                bowlers_in_xi.append({'name':p['name'],'wickets':p.get('wickets',0),'econ':p.get('econ',10)})
                added.add(p['name'])
            if len(bowlers_in_xi)>=NUM_BOWL: break
    while len(bowlers_in_xi)<NUM_BOWL:
        bowlers_in_xi.append({'name':'','wickets':0,'econ':0})
    tw_lookup[(team,year)]=bowlers_in_xi[:NUM_BOWL]

print(f"  Rosters: {len(full_roster)}", flush=True)

# Form tracker
bs_sorted=bs.sort_values(['batsman','date']).reset_index(drop=True)
bw_sorted=bw.sort_values(['bowler','date']).reset_index(drop=True)
bat_form={}
for p,grp in bs_sorted.groupby('batsman'):
    grp=grp.reset_index(drop=True)
    for i in range(len(grp)):
        prev=grp.iloc[max(0,i-5):i]; mid=grp.loc[i,'mid']
        bat_form[(mid,p)]={'fa':round(prev['runs'].sum()/len(prev),1) if len(prev)>0 else 0,
                          'fs':round(prev['runs'].sum()/prev['balls'].sum()*100,1) if len(prev)>0 and prev['balls'].sum()>0 else 0}
bowl_form={}
for p,grp in bw_sorted.groupby('bowler'):
    grp=grp.reset_index(drop=True)
    for i in range(len(grp)):
        prev=grp.iloc[max(0,i-5):i]; mid=grp.loc[i,'mid']
        if len(prev)==0: bowl_form[(mid,p)]={'fw':0,'fe':0}
        else: fw=prev['wkts'].sum();fr=prev['conc'].sum();fb=prev['balls'].sum(); bowl_form[(mid,p)]={'fw':fw,'fe':round(fr/(fb/6),2) if fb>0 else 0}

h2h=df.groupby(['batsman','bowler','mid']).agg(runs=('striker','sum'),dismissals=('wb','sum')).reset_index()
h2ha=h2h.groupby(['batsman','bowler']).agg(runs=('runs','sum'),dismissals=('dismissals','sum')).reset_index()
h2h_lookup={}
for _,r in h2ha.iterrows(): h2h_lookup[(r['batsman'],r['bowler'])]={'r':r['runs'],'d':r['dismissals']}

bm_map={}; [bm_map.update({(r['mid'],r['batsman']):{'runs':r['runs'],'balls':r['balls']}}) for _,r in bs.iterrows()]
bwm_map={}; [bwm_map.update({(r['mid'],r['bowler']):{'wickets':r['wkts'],'runs':r['conc'],'balls':r['balls']}}) for _,r in bw.iterrows()]

# ============================================================
# BUILD FEATURES
# ============================================================
print("BUILDING FEATURES", flush=True)
dfm=df[df['overs']>=5.0].copy()
total=len(dfm); t0=time.time()

# Base features (9) + player features (11*6 + 5*6 = 96) + team OHE
n_base=9  # runs,wickets,overs,runs_last_5,wickets_last_5,run_rate,balls_bowled,wickets_left,venue_avg
n_bat=NUM_BAT*6 + 3*2  # 72: 11*6 main + 3 top batsmen form (fa, fs)
n_bowl=NUM_BOWL*6 + 2*2  # 34: 5*6 main + 2 top bowlers form (fw, fe)
n_extra=4  # venue_avg_bat_team, is_home, depth_runs, depth_count
n_cols=n_base+n_bat+n_bowl+n_extra

X_arr=np.zeros((total,n_cols),dtype=np.float64)
bat_names=np.empty(total,dtype=object)
bowl_names=np.empty(total,dtype=object)
totals_arr=np.zeros(total,dtype=np.float64)
years_arr=np.zeros(total,dtype=np.int64)

for idx in range(total):
    row=dfm.iloc[idx]
    mid=row['mid']; yr=int(row['year']); bat=row['bat_team']; bowl=row['bowl_team']
    top=tb_lookup.get((bat,yr),[]); topw=tw_lookup.get((bowl,yr),[])
    ob=row['overs']; bballs=int(ob)*6+round((ob-int(ob))*10)
    va_fill=va.get(row['venue_clean'],165)
    
    # Base [0:9]
    X_arr[idx,0]=row['runs']; X_arr[idx,1]=row['wickets']; X_arr[idx,2]=ob
    X_arr[idx,3]=row['runs_last_5']; X_arr[idx,4]=row['wickets_last_5']
    X_arr[idx,5]=row['runs']/ob if ob>0 else 0
    X_arr[idx,6]=bballs; X_arr[idx,7]=10-row['wickets']; X_arr[idx,8]=va_fill
    bat_names[idx]=bat; bowl_names[idx]=bowl; totals_arr[idx]=row['total']; years_arr[idx]=yr        # Extra features [115:119]
        X_arr[idx,115]=va_bt.get((bat,row['venue_clean']),va_fill)
        X_arr[idx,116]=row['is_home']

        # Depth features: remaining batting strength
        remaining_runs = 0
        remaining_count = 0
        for b in top:
            if b['name'] and (mid, b['name']) not in bm_map:
                remaining_runs += b.get('runs', 0)
                remaining_count += 1
        X_arr[idx,117] = remaining_runs
        X_arr[idx,118] = remaining_count

        # Batting [9:81]
    for i in range(NUM_BAT):
        bi=9+min(i,3)*2+i*6  # first 3 slots have 8 cols, rest have 6
        if i<len(top) and top[i]['name']:
            b=top[i]; ply=(mid,b['name']) in bm_map
            X_arr[idx,bi]=b['runs']; X_arr[idx,bi+1]=b['avg']; X_arr[idx,bi+2]=b['sr']
            X_arr[idx,bi+3]=1 if ply else 0                if ply:
                    mm=bm_map[(mid,b['name'])]
                    X_arr[idx,bi+4]=mm['runs']; X_arr[idx,bi+5]=mm['balls']
            # Add form for top 3 batsmen (in extra 3 cols after main 6)
            if i < 3:
                fmt=bat_form.get((mid,b['name']),{})
                X_arr[idx,bi+6]=fmt.get('fa',0); X_arr[idx,bi+7]=fmt.get('fs',0)

    # Bowling [81:115]
    for i in range(NUM_BOWL):
        wi=81+min(i,2)*2+i*6  # first 2 slots have 8 cols, rest have 6
        if i<len(topw) and topw[i]['name']:
            b=topw[i]; ply=(mid,b['name']) in bwm_map
            X_arr[idx,wi]=b['wickets']; X_arr[idx,wi+1]=b['econ']
            X_arr[idx,wi+2]=1 if ply else 0
            if ply: mm=bwm_map[(mid,b['name'])]; X_arr[idx,wi+3]=mm['wickets']; X_arr[idx,wi+4]=mm['runs']; X_arr[idx,wi+5]=mm['balls']
            # Add form for top 2 bowlers
            if i < 2:
                fmt=bowl_form.get((mid,b['name']),{})
                X_arr[idx,wi+6]=fmt.get('fw',0); X_arr[idx,wi+7]=fmt.get('fe',0)

    if (idx+1)%100000==0: print(f"  {idx+1}/{total} ({(idx+1)/(time.time()-t0):.0f} r/s)", flush=True)

base_cols=['runs','wickets','overs','runs_last_5','wickets_last_5','run_rate','balls_bowled','wickets_left','venue_avg']
extra_cols=['venue_avg_bat_team','is_home','depth_runs','depth_count']
bat_cols=[]; bowl_cols=[]
for i in range(NUM_BAT):
    for s in ['runs_season','avg','sr','playing','match_runs','match_balls']:
        bat_cols.append(f'top_bat_{i+1}_{s}')
    if i < 3:  # Form for top 3 batsmen
        bat_cols.append(f'top_bat_{i+1}_form_avg')
        bat_cols.append(f'top_bat_{i+1}_form_sr')
for i in range(NUM_BOWL):
    for s in ['wickets_season','econ','playing','match_wickets','match_runs','match_balls']:
        bowl_cols.append(f'top_bowl_{i+1}_{s}')
    if i < 2:  # Form for top 2 bowlers
        bowl_cols.append(f'top_bowl_{i+1}_form_wkts')
        bowl_cols.append(f'top_bowl_{i+1}_form_econ')

feat=pd.DataFrame(X_arr,columns=base_cols+bat_cols+bowl_cols+extra_cols)
feat['bat']=bat_names; feat['bowl']=bowl_names; feat['total']=totals_arr; feat['year']=years_arr
del X_arr,bat_names,bowl_names,totals_arr,years_arr
import gc; gc.collect()
print(f"  Done: {len(feat):,} x {len(feat.columns)}", flush=True)

# ============================================================
# TRAIN
# ============================================================
print("TRAINING", flush=True)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

all_teams=sorted(set(df['bat_team'].unique())|set(df['bowl_team'].unique()))
xc=base_cols+bat_cols+bowl_cols+extra_cols
fn=list(xc)
for t in all_teams: fn.append(f'bat_team_{t}')
for t in all_teams: fn.append(f'bowl_team_{t}')

def build_X(mask, data):
    n=int(mask.sum()); nf=len(xc)+len(all_teams)*2
    X=np.zeros((n,nf),dtype=np.float64)
    sub=data[mask]
    X[:,:len(xc)]=sub[xc].values
    bats=sub['bat'].values; bowls=sub['bowl'].values
    for i,t in enumerate(all_teams):
        X[bats==t,len(xc)+i]=1
        X[bowls==t,len(xc)+len(all_teams)+i]=1
    return X

rng=np.random.RandomState(42)
idx_all=np.arange(len(feat))
rng.shuffle(idx_all)
n_total=len(feat)
n_train=int(n_total*0.8)
tr_idx=idx_all[:n_train]; te_idx=idx_all[n_train:]
tr=np.zeros(len(feat),dtype=bool); tr[tr_idx]=True
te=np.zeros(len(feat),dtype=bool); te[te_idx]=True
# Cap training to 100k for speed
if tr.sum()>100000:
    keep=rng.choice(tr_idx,100000,replace=False)
    tr_arr=np.zeros(len(feat),dtype=bool); tr_arr[keep]=True; tr=tr_arr

print(f"  Train: {tr.sum():,} (all years 2008-2026), Test: {te.sum():,}", flush=True)
print(f"  Building X...", flush=True)
Xt=build_X(tr,feat); yt=feat.loc[tr,'total'].values
Xe=build_X(te,feat); ye=feat.loc[te,'total'].values
print(f"  X train: {Xt.shape}, X test: {Xe.shape}", flush=True)
del feat; gc.collect()

print(f"  Training GBR...", flush=True)
gb=GradientBoostingRegressor(n_estimators=100,max_depth=6,learning_rate=0.05,subsample=0.5,min_samples_leaf=5,random_state=42,verbose=0)
gb.fit(Xt,yt)
print(f"  GBR done", flush=True)
print(f"  Train MAE: {mean_absolute_error(yt,gb.predict(Xt)):.2f}", flush=True)
print(f"  Test MAE: {mean_absolute_error(ye,gb.predict(Xe)):.2f}", flush=True)

if mw:
    print(f"  Win model...", flush=True)
    cr=[]
    for mid_val in mw:
        mdf=df[df['mid']==mid_val]
        for _,r2 in mdf.iterrows():
            o=r2['overs']
            if o<5 or o>=19.5: continue
            fi=r2['total']; r=r2['runs']; w=r2['wickets']; rl=fi-r; bal=(20-o)*6
            if bal<=0: continue
            win=1 if mw[mid_val]==r2['bat_team'] else 0
            cr.append({'target':fi,'ol':20-o,'rl':rl,'wl':10-w,'rrr':(rl/bal)*6,
                       'crr':r/o if o>0 else 0,'rd':(r/o if o>0 else 0)-((rl/bal)*6),
                       'rl5':r2['runs_last_5'],'wl5':r2['wickets_last_5'],'win':win})
    if cr:
        cd=pd.DataFrame(cr)
        wf=['target','ol','rl','wl','rrr','crr','rd','rl5','wl5']
        wm=RandomForestClassifier(100,random_state=42,max_depth=8).fit(cd[wf],cd['win'])
        print(f"  Win acc: {accuracy_score(cd['win'],wm.predict(cd[wf]))*100:.1f}% ({len(cd)} samples)", flush=True)
    else: wm=None
else: wm=None

# ============================================================
# ENSEMBLE: GBR + Ridge regression
# ============================================================
print("ENSEMBLE", flush=True)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=50, random_state=42, max_iter=2000)
ridge.fit(Xt, yt)
ridge_train_mae = mean_absolute_error(yt, ridge.predict(Xt))
ridge_test_mae = mean_absolute_error(ye, ridge.predict(Xe))
print(f"  Ridge Train MAE: {ridge_train_mae:.2f}, Test MAE: {ridge_test_mae:.2f}", flush=True)

# Find optimal blend weight on test set
from sklearn.metrics import r2_score
alphas = np.arange(0, 1.01, 0.025)
blend_scores = []
gb_preds = gb.predict(Xe)
ridge_preds = ridge.predict(Xe)
blend_results = []
for a in alphas:
    blended = a * gb_preds + (1-a) * ridge_preds
    m = mean_absolute_error(ye, blended)
    r2 = r2_score(ye, blended)
    blend_results.append((a, m, r2))
best_idx = np.argmin([r[1] for r in blend_results])
best_alpha = blend_results[best_idx][0]
best_mae = blend_results[best_idx][1]
best_r2 = blend_results[best_idx][2]
print(f"  Optimal blend: GBR weight={best_alpha:.3f}, Ridge weight={1-best_alpha:.3f}", flush=True)
print(f"  Ensemble Test MAE: {best_mae:.2f}, R2: {best_r2:.3f}", flush=True)

# Compare: GBR alone vs Ridge alone vs Ensemble vs CRR blend
proj_mae = mean_absolute_error(ye, gb_preds)
print(f"\n  Summary:")
print(f"    GBR alone:      MAE={proj_mae:.2f}")
print(f"    Ridge alone:    MAE={ridge_test_mae:.2f}")
print(f"    Ensemble:       MAE={best_mae:.2f}")
print(f"    GBR weight:     {best_alpha:.3f}")
print(f"    Ridge weight:   {1-best_alpha:.3f}")
print(f"    Train size:     {len(yt):,}, Test size: {len(ye):,}", flush=True)

# Save ensemble info for use at inference
ensemble_info = {
    'ridge_model': ridge,
    'gbr_weight': best_alpha,
    'ridge_weight': 1 - best_alpha,
    'ensemble_mae': best_mae,
    'ridge_mae': ridge_test_mae,
}

# ============================================================
# SAVE
# ============================================================
print("SAVING", flush=True)
tsp={}
for key,roster in full_roster.items():
    parts=key.split('|'); team=parts[0]; year=int(parts[1])
    xi_names=roster.get('xi_names',[])
    top5=[]; added=set()
    for n in xi_names:
        p=roster['all_players'].get(n,{})
        if p and 'runs' in p:
            if p['name'] not in added:
                top5.append(p); added.add(p['name'])
    for p in sorted(roster['batsmen'],key=lambda x:-x['runs']):
        if len(top5)>=5: break
        if p['name'] not in added: top5.append(p); added.add(p['name'])
    top3=[]; added2=set()
    for n in xi_names[-5:]:
        p=roster['all_players'].get(n,{})
        if p and 'wickets' in p:
            if p['name'] not in added2:
                top3.append(p); added2.add(p['name'])
    for p in sorted(roster['bowlers'],key=lambda x:-x['wickets']):
        if len(top3)>=3: break
        if p['name'] not in added2: top3.append(p); added2.add(p['name'])
    tsp[key]={
        'top_batsmen':top5,'top_bowlers':top3,
        'xi_names':roster.get('xi_names',[]),
        'all_players':sorted(roster['all_players'].values(),key=lambda x:-x.get('matches',0))
    }

years_avail=sorted(df['year'].unique().astype(int))
venues=sorted(df['venue_clean'].unique())

output={
    'gb_model':gb,'feature_names':fn,'orig_features':base_cols,'player_features':bat_cols+bowl_cols,
    'bat_teams_enc':all_teams,'bowl_teams_enc':all_teams,
    'win_model':wm,'win_features':wf if cr else [],
    'team_season_players':tsp,'full_roster':full_roster,'venue_avg_scores':va,
    'venues':venues,'all_teams':all_teams,'years_available':years_avail,
    'num_bat_slots':NUM_BAT,'num_bowl_slots':NUM_BOWL,
    'ensemble':ensemble_info
}
with open(MODEL_OUT,'wb') as f: pickle.dump(output,f)
print(f"  {len(fn)} feats, {len(tsp)} team-seasons", flush=True)
print("DONE", flush=True)
