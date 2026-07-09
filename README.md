# 🏏 IPL Predictor Pro

**Real-time IPL score prediction & win probability estimator** using ball-by-ball data from 2008–2026 (203,773 rows, 1,276 match-innings). Features an 11-player XI system with per-player season stats, form tracking, venue context, and interactive match state inputs.

Unlike typical cricket predictors that use only match-level aggregates (current score, wickets, overs), this model reasons about **who** is batting and bowling. By encoding each of the 11 players with season + match + form features, the model learns that *Virat Kohli at the crease* is very different from *a No. 8 batsman* — even at the same score.

---

## The Problem

IPL is unpredictable because team composition matters as much as match state. Two teams at 80/2 in the 10th over can have wildly different projected scores depending on whether the batting XI has deep batting or the bowling attack has death-over specialists.

**Traditional approaches fail** because they use:
- Current score, wickets, overs only (no player identity)
- Simple run-rate extrapolation (doesn't account for wicket quality or batting depth)
- Team-level averages (lose per-player strength differences)

**This approach solves it** with a full 11-player XI encoding:

| Slot | What the model "sees" |
|------|----------------------|
| Batsman 1 | Season runs, avg, SR, playing?, match runs/balls, form avg/SR |
| Batsman 2 | Same — even if they haven't batted yet, the model knows their capability |
| ... | ... |
| Bowler 1 | Season wickets, economy, playing?, match wkts/runs/balls, form wkts/econ |
| Bowler 2 | Same — the model knows who's still available to bowl |

---

## Features

- **Score Prediction** — Gradient Boosting Regressor predicts final innings total from current match state
- **Win Probability** — Random Forest Classifier estimates chase success likelihood
- **Quick & Advanced Modes** — Quick uses season averages; Advanced accepts per-player match stats
- **Full XI Editor** — Pick any 11 players per team-season, choose 5 bowlers from the XI
- **Auto-Calc from Match Stats** — Enter each batsman's runs/balls and bowler's figures; prediction auto-calculates
- **Interactive Charts** — Run contribution bar chart, bowling figures, chase profile projection
- **Player Stats Browser** — Browse top performers for any team-season combination
- **153 Features** — 9 base match-state + 72 batting (11 slots) + 34 bowling (5 slots) + 4 venue/depth context + team OHE
- **Feature Importance Analysis** — Built-in breakdown with per-slot and per-category importance charts

---

## Quick Start

### Clone, Install & Run

```bash
# Clone the repo
git clone https://github.com/batramayank106/IPL-Score-And-Chase-Predictor.git
cd IPL-Score-And-Chase-Predictor

# Install dependencies
pip install -r requirements.txt

# Run the app (model included — no training needed)
streamlit run streamlit_app.py
```

A pre-trained model (`ipl_models_enhanced.pkl`) is included in the repo. No training required — just install and run.

### You Can Also Try To Improve/Retrain the Model

The training pipeline is in [`train_enhanced_model.py`](train_enhanced_model.py). You can modify it and retrain:

```bash
# 1. Edit train_enhanced_model.py to experiment with:
#    - New features (add more player stats, venue context, etc.)
#    - Different model architectures (XGBoost, LightGBM, neural nets)
#    - Hyperparameter tuning (n_estimators, max_depth, learning_rate)
#    - Different ensemble blends (adjust grid search in the ENSEMBLE section)

# 2. Retrain with your changes
python train_enhanced_model.py

# 3. The new model overwrites ipl_models_enhanced.pkl — launch the app
streamlit run streamlit_app.py
```

Key sections you can explore in `train_enhanced_model.py`:
- **Feature engineering** (lines ~252–348) — add/modify columns in the feature matrix
- **GBR training** (line ~394) — tweak `GradientBoostingRegressor` parameters
- **Ensemble blend** (lines ~433–460) — change the grid search range or try different regressors
- **Ridge model** (line ~427) — adjust `alpha` or swap in another linear model
- **Win model** (lines ~400–420) — modify the `RandomForestClassifier` for chase prediction

After retraining, test your improvements with:

```bash
python test/test_ensemble_weights.py   # Tests all GBR/Ridge blend ratios
python test/test_pred.py               # Runs sample predictions
```

### Train Locally (Optional)

```bash
pip install -r requirements.txt
python train_enhanced_model.py   # takes ~5 minutes
streamlit run streamlit_app.py
```

### Train in Google Colab (Optional)

Open [`ipl_predictor_colab.ipynb`](ipl_predictor_colab.ipynb), upload `ipl.csv`, run all cells. Download the resulting `ipl_models_enhanced.pkl`.

---

## Project Structure

```
IPL-Score-And-Chase-Predictor/
├── streamlit_app.py            # Main Streamlit UI
├── train_enhanced_model.py     # Data pipeline + training (153 features)
├── build_player_stats.py       # Build player_stats.csv from ipl.csv
├── ipl_predictor_colab.ipynb   # Google Colab notebook
├── requirements.txt            # Python dependencies
├── ipl.csv                     # Ball-by-ball IPL data (203,773 rows, 1,276 mids, 2008–2026)
├── player_stats.csv            # Per-player season stats (3,200 rows, 780 players, 2008–2026)
├── ipl_models_enhanced.pkl     # Pre-trained V2 model (~5 MB)
├── screenshots/                # UI screenshots
├── examples/                   # Usage examples
├── test/                       # Test scripts
├── .gitignore
└── README.md
```

---

## How It Works

### Data Pipeline

1. **Load & Clean** — Ball-by-ball IPL data from 2008–2026
2. **Player Stats** — Season aggregates per batsman/bowler (runs, avg, SR, wickets, economy)
3. **Role Detection** — Classify each player as bat/AR/bowl based on batting vs bowling appearances
4. **Auto XI** — Build an optimal 11 based on roles, sorted by batting position
5. **Form Features** — Last 5 innings form for top 3 batsmen (avg, SR) and top 2 bowlers (wkts, economy)
6. **Venue Context** — Venue average score + team-specific venue average + home advantage flag
7. **Depth Features** — Remaining batting strength (season runs of XI players who haven't batted yet)
8. **Feature Matrix** — 153 features
9. **Train** — GradientBoostingRegressor (score) + RandomForestClassifier (win %)

### Feature Layout (153 Total)

| Category | Count | Description |
|----------|-------|-------------|
| Match State | 9 | runs, wickets, overs, runs_last_5, wkts_last_5, run_rate, balls_bowled, wkts_left, venue_avg |
| Batting slots | 72 (11 slots × 6 + 3×2 form) | runs_season, avg, sr, playing, match_runs, match_balls + form_avg, form_sr (top 3) |
| Bowling slots | 34 (5 slots × 6 + 2×2 form) | wickets_season, econ, playing, match_wickets, match_runs, match_balls + form_wkts, form_econ (top 2) |
| Venue/Depth | 4 | venue_avg_bat_team, is_home, depth_runs, depth_count |
| Team OHE | 34 | bat_team_* + bowl_team_* (one-hot encoded for all 17 teams) |

### Model Performance (V2)

| Metric | Original (V1) | Current (V2) | Improvement |
|--------|---------------|--------------|-------------|
| Features | 151 | 153 | +2 depth features |
| GBR max_depth | 4 | 6 | Deeper trees |
| GBR Test MAE | 13.23 | 9.74 | -26.4% |
| Ensemble Test MAE | 12.65 | 9.74 | -23.0% |
| R² | 0.706 | 0.831 | +0.125 |
| **Ensemble Weights** | **GBR 50% + Ridge 50%** | **GBR 100% + Ridge 0%** | Ridge confirmed 0% (grid search over 101 blend ratios) |
| Win Accuracy | 69.2% | 69.2% | Same win model |
| Training Data | 100k samples | 100k samples | Same |
| Test Data | 40,755 samples | 40,755 samples | Same |

---

## Feature Engineering

### Per-Player Slot Features

Each of the 11 batting slots and 5 bowling slots gets 6 main features, with extra form features for the top 3 batsmen and top 2 bowlers:

| Feature | Source | Why it matters |
|---------|--------|----------------|
| `runs_season` / `wickets_season` | Season aggregate | Overall batting/bowling quality |
| `avg` / `econ` | Season per-match average/economy | Consistency |
| `sr` | Season strike rate | Scoring rate capability |
| `playing` | Is this player currently active? | Active contribution vs potential |
| `match_runs` / `match_wickets` | Runs/wickets so far | Current innings impact |
| `match_balls` | Balls faced/bowled | Settled vs new at crease, overs bowled |
| `form_avg` / `form_wkts` | Last 5 innings avg/wickets | Recent form (top 3 bat, top 2 bowl only) |
| `form_sr` / `form_econ` | Last 5 innings SR/economy | Recent scoring/bowling intent |

**V2 change**: All 11 batting positions now use `match_runs` and `match_balls` (positions 8–11 are no longer zeroed out). The model captures batting depth more naturally through these features plus the two new depth features below.

### Depth Features (V2 New)

| Feature | Description |
|---------|-------------|
| `depth_runs` | Sum of season runs for XI players who haven't batted yet |
| `depth_count` | Count of XI players who haven't batted yet |

These features capture remaining batting strength — a team at 80/2 with Virat Kohli still to bat has very different depth than one at 80/2 with only bowlers remaining.

### Venue Context

| Feature | Description |
|---------|-------------|
| `venue_avg` | Average total at this venue (death overs data) |
| `venue_avg_bat_team` | Average total for this batting team at this specific venue |
| `is_home` | 1 if the batting team is playing at their home city, 0 otherwise |

### Team Encoding

Teams are one-hot encoded (batting team + bowling team) with all 17 historical IPL teams. While individual team importance is low (~0.3% total), the team signal is captured through player-specific features and the team-venue interaction in `venue_avg_bat_team`.

---

## Feature Importance Analysis

The app includes a built-in feature importance breakdown under the "Model Feature Importances" expander. Current V2 model importances:

### Category Breakdown

| Category | Total Importance | Key Contributors |
|----------|----------------|-----------------|
| Match State | ~58.85% | Run Rate (55.95%), Venue × Team Avg (6.70%), Total Runs (2.90%) |
| Bowler Features | ~15.65% | Bowl1-5 match_runs (5.42%–0.86%), economies, season wickets |
| Batting Features | ~10.40% | Bat1-11 match_runs, SR, avg, form |
| Team OHE | ~4.80% | One-hot encoded team indicators |

### How match_runs importance differs by position

| Bat | Importance | Notes |
|-----|-----------|-------|
| Bat1 | 1.75% | Opener's current runs |
| Bat2 | 1.20% | Other opener's runs |
| Bat3 | 2.05% | Highest — #3 is typically the best batter |
| Bat4 | 1.17% | Middle order |
| Bat5 | 1.41% | Middle order |
| Bat6 | 0.80% | Lower middle |
| Bat7 | 0.65% | Tail start |
| Bat8-11 | ~0.5-1.0% | Low but non-zero — captures lower-order contributions (V2) |

---

## Model Architecture

### Score Predictor: Gradient Boosting Regressor

```
GBR: n_estimators=100, learning_rate=0.05, max_depth=6, min_samples_leaf=5, subsample=0.5
Ridge: alpha=50, max_iter=2000 (kept for reference but weight confirmed 0%)
Ensemble: GBR 100% — grid search over all blend weights (0%–100% in 1% steps) confirmed pure GBR is optimal
```

| Model | Test MAE |
|-------|----------|
| GBR alone | 9.74 |
| Ridge alone | 13.07 |
| **Ensemble (GBR 100%)** | **9.74** |
| Ensemble R² | 0.831 |

Verified via grid search over 101 blend ratios on 40,755 holdout samples. **Why can't Ridge contribute anything?** Ridge is a linear model — it sees a team at 80/2 and predicts the same total regardless of whether Virat Kohli or a No. 8 batsman is at the crease. GBR's trees learn interactions naturally: Kohli at slot 3 with 2 wickets down gets a very different projection than the same score with 8 wickets down. Since Ridge cannot capture *who* is batting and *which* bowlers are left, its predictions are consistently 32% worse (MAE 13.07 vs 9.74). Blending it in at any ratio only drags the ensemble toward Ridge's errors. Pure GBR is optimal.

### Win Probability Classifier: Random Forest

```
n_estimators=100
max_depth=8
random_state=42
```

- **Accuracy**: 69.2% — percentage of chase situations where the model correctly predicted the winner, trained on 176k mid-innings states from 2008–2026
- Uses current match state features (target, overs left, runs left, wickets left, required RR, current RR, RR difference, last 5 overs data)

### Data Split

- **Train**: 100,000 random samples (80% of ball-by-ball rows from 2008–2026)
- **Test**: 40,755 random samples (20% holdout from all seasons)
- Random split ensures the model sees high-scoring modern matches during training

### Dataset Stats (Current)

| Metric | Value |
|--------|-------|
| Ball-by-ball rows | 203,773 |
| Match-innings (mids) | 1,276 |
| Unique players | 780 |
| Player-season records | 3,200 |
| Teams | 19 |
| Seasons | 2008–2026 |

---

## Post-Processing & Recalibration

The raw model output is enhanced with post-processing to produce more realistic projections:

### CRR Blend
When the batting side has wickets in hand and a healthy scoring rate, the prediction blends toward a current-run-rate-based projection. This accounts for the model being trained on historical data where scoring was generally lower than modern T20 standards.

### Wicket Adjustment
- **Depth intact** (0-2 wickets down): Boost prediction to reflect top-order scoring potential
- **Exposed tail** (4+ wickets down): Penalize prediction to reflect expected scoring drop-off beyond the top 3

### Recalibration Approach
The V2 model was validated against the original using 6,926 real match prediction points. Post-processing parameters were optimized via grid search over wicket adjustment multipliers and CRR blend factors. The key findings:

- V2 raw predictions (no post-processing): MAE 28.92 (6.1% better than original)
- V2 with calibrated wicket adjustment: MAE 24.92 (19.1% improvement)
- Grid-search optimal parameters: MAE 22.78 (26.6% improvement)
- Batting depth features (`depth_runs`, `depth_count`) provide genuine signal for remaining batting strength

---

## How to Use the App

### Match Predictor Tab

1. Select **Mode** — "First Innings Score" or "Chase Win Probability"
2. Select **Detail Level** — "Quick (season avg)" or "Advanced (match input)"
3. Pick **teams, year, venue** — batting and bowling teams can use different seasons
4. **Quick mode** — Auto-generated XI shown read-only
5. **Advanced mode** — Edit the XI (11 players), pick bowlers, enter match stats per player
6. Click **Predict** — see projected score, runs contribution chart, bowling figures

### Player Stats Tab

- Browse any team-season to see all players with season stats (runs, avg, SR, wickets, econ, role)
- Sort by runs, wickets, avg, SR, or economy
- Visual chart of top 15 players by selected metric
- Summary stats: total players, batters with runs, bowlers with wickets, top batter
- Data sourced from `player_stats.csv` (777 players, 2008–2026)

### Feature Importance (Built-in)

- Expand "Model Feature Importances" at the bottom to see:
  - Top 20 feature importances bar chart
  - Category-level breakdown (Match State, Batting, Bowling, Team)
  - Per-position match runs importance
  - Per-position season runs importance
  - Explanatory insights for each chart

---

## Live Demo

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ipl-score-and-chase-predictor-06.streamlit.app)

The app is live on **Streamlit Community Cloud**.  
Visit: [https://ipl-score-and-chase-predictor-06.streamlit.app](https://ipl-score-and-chase-predictor-06.streamlit.app)

---

## Datasets

### `ipl.csv` — Ball-by-Ball Data

203,773 rows covering 1,276 match-innings from 2008–2026 (19 teams). Each row represents one ball with match context.

Expected columns:
```
mid, date, venue, bat_team, bowl_team, batsman, bowler, runs,
wickets, overs, runs_last_5, wickets_last_5, striker, non-striker, total
```

Available on [Kaggle](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set).

### `player_stats.csv` — Player Season Stats

3,200 rows covering 780 unique players across 2008–2026. Pre-computed per-player season aggregates for the Player Stats tab.

Columns: `player, year, team, runs, avg, sr, wickets, econ, role`

Regenerate with:
```bash
python build_player_stats.py
```

---

## Screenshots

<p align="center">
  <img src="screenshots/Main Interface of quick mode - First Innings.png" alt="Quick Mode — First Innings" width="600"/>
  <br><em>Quick Mode — First Innings Score Prediction</em>
</p>

<p align="center">
  <img src="screenshots/Advanced Mode -  Layout.png" alt="Advanced Mode Layout" width="600"/>
  <br><em>Advanced Mode — Full XI Editor & Player Match Stats</em>
</p>

<p align="center">
  <img src="screenshots/First Run - Quick Mode First Innings.png" alt="First Innings Prediction Result" width="600"/>
  <br><em>First Innings — Smart Projection with CRR Blend</em>
</p>

<p align="center">
  <img src="screenshots/Fist Run - Quick Mode Chase Analysis.png" alt="Chase Win Probability" width="600"/>
  <br><em>Chase Win Probability with Chase Profile Chart</em>
</p>

<p align="center">
  <img src="screenshots/Player Stats.png" alt="Player Stats" width="600"/>
  <br><em>Player Stats Tab — Browse Any Team-Season</em>
</p>

<p align="center">
  <img src="screenshots/Model Importance Graphs.png" alt="Feature Importance" width="600"/>
  <br><em>Built-in Feature Importance Analysis — Per-Slot & Category Breakdown</em>
</p>

<p align="center">
  <img src="screenshots/Advanced Mode - First Run - First Innings.png" alt="Advanced Mode First Innings" width="600"/>
  <br><em>Advanced Mode — First Innings Prediction with Match Stats</em>
</p>

<p align="center">
  <img src="screenshots/Advanced Mode - Showing Warnings of Runs and Bowls Mismatch between Batsman and Bowler.png" alt="Validation Warnings" width="600"/>
  <br><em>Advanced Mode — Input Validation (runs/balls mismatch warnings)</em>
</p>

<p align="center">
  <img src="screenshots/Avanced Mode - Second Run - Second Innings.png" alt="Advanced Mode Second Innings" width="600"/>
  <br><em>Advanced Mode — Second Innings Prediction</em>
</p>

<p align="center">
  <img src="screenshots/Second Run - Quick Mode Chase.png" alt="Second Chase Run" width="600"/>
  <br><em>Quick Mode — Second Chase Prediction</em>
</p>

<p align="center">
  <img src="screenshots/Second Run - Quick Mode Second Innings.png" alt="Second Innings Quick" width="600"/>
  <br><em>Quick Mode — Second Innings Prediction</em>
</p>

<p align="center">
  <img src="screenshots/Third Run - Quick Mode Chase.png" alt="Third Chase Run" width="600"/>
  <br><em>Quick Mode — Third Chase Prediction</em>
</p>

---

## 👨‍💻 Author

**Mayank Batra** — Student at NIT Warangal

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank%20Batra-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/mayank-batra-821b10365/)

---

## 📄 License

MIT
