# 4_Schedule_Predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Schedule Predictor")


# ---------------------------
# Disclaimer / Note
# ---------------------------
st.markdown("""
###
⚠️ **Important Note:**  
The schedules and matchups shown here are **randomly generated**.  
They are designed to highlight the structure of predicted qualities and outputs within the model framework.  
The bulk of the underlying predictive work remains proprietary and is held as a **competitive advantage**.  
This page provides **slight examples of the coding logic** used without revealing too much detail, but it is included here because it ties the full system together.
Some details on this page might be inaccurate with the random generator being connected to this sheet.

""")


# -----------------------
# Load data helpers
# -----------------------
@st.cache_data
def load_all_stats(path="Data/All_stats.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, encoding="latin1")
    return df

@st.cache_data
def load_history(path="Data/Daily_predictor_excel.csv"):
    if not os.path.exists(path):
        st.warning(f"{path} not found — historical training disabled.")
        return None
    df = pd.read_csv(path, encoding="latin1")
    return df

@st.cache_data
def load_schedule(path="Data/Randomized_Schedule.csv"):
    if not os.path.exists(path):
        st.info(f"{path} not found — schedule will be built from All_stats (simple fallback).")
        return None
    df = pd.read_csv(path, encoding="latin1")
    return df

# Load files
try:
    df_all = load_all_stats()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

df_hist = load_history()
schedule_df = load_schedule()

# Normalize column names (safer lookups)
# Note: All_stats uses "Teams" according to your data sample
if "Teams" not in df_all.columns and "Team" in df_all.columns:
    df_all = df_all.rename(columns={"Team": "Teams"})

# history sample uses "Team" and "Opponent"
# no rename here, we'll reference both names directly
st.sidebar.markdown("## Data files loaded")
st.sidebar.write({
    "All_stats rows": len(df_all),
    "History rows": len(df_hist) if df_hist is not None else None,
    "Schedule rows": len(schedule_df) if schedule_df is not None else None,
})

# -----------------------
# Prepare training data from history
# -----------------------
# Helpers to detect home team and scores in history
def detect_home_away_and_scores(hist):
    """
    Returns a DataFrame with columns: home_team, away_team, home_score, away_score.
    Heuristics:
      - If 'Road Game' column exists and is 1/0: if Road Game==1 then Team traveled -> opponent was home.
      - Else if 'Location' contains a city like 'Lawrence, KS' with the Team's school location missing,
        we fallback to treating Team as home unless 'Road Game' says otherwise.
      - Expects 'Points' and 'Opp Points' columns for scores.
    """
    h = hist.copy()
    # standardize name columns
    team_col = None
    opp_col = None
    if "Team" in h.columns:
        team_col = "Team"
    elif "Teams" in h.columns:
        team_col = "Teams"
    if "Opponent" in h.columns:
        opp_col = "Opponent"
    # scores
    score_col = None
    opp_score_col = None
    if "Points" in h.columns and "Opp Points" in h.columns:
        score_col = "Points"; opp_score_col = "Opp Points"
    elif "PTS" in h.columns and "OPP_PTS" in h.columns:
        score_col = "PTS"; opp_score_col = "OPP_PTS"

    if team_col is None or opp_col is None or score_col is None:
        return None  # not enough info

    # Road Game detection
    road_col = None
    for c in h.columns:
        if c.strip().lower() in ("road game", "road", "is_road", "is_away"):
            road_col = c
            break

    rows = []
    for idx, r in h.iterrows():
        team = r[team_col]
        opp = r[opp_col]
        team_score = pd.to_numeric(r.get(score_col, np.nan), errors="coerce")
        opp_score = pd.to_numeric(r.get(opp_score_col, np.nan), errors="coerce")

        # default: assume Team is home unless road marker says otherwise or Location suggests otherwise
        is_team_road = False
        if road_col is not None:
            try:
                val = int(pd.to_numeric(r.get(road_col, 0), errors="coerce") or 0)
                is_team_road = (val == 1)
            except Exception:
                is_team_road = False

        if is_team_road:
            home = opp
            away = team
            home_score = opp_score
            away_score = team_score
        else:
            home = team
            away = opp
            home_score = team_score
            away_score = opp_score

        rows.append({
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score
        })
    return pd.DataFrame(rows)

# Build training dataframe if possible
model = None
feature_cols = None
train_warning = None

if df_hist is not None:
    hist_parsed = detect_home_away_and_scores(df_hist)
    if hist_parsed is None:
        train_warning = ""
    else:
        # attach target
        hist_parsed["home_win"] = (hist_parsed["home_score"] > hist_parsed["away_score"]).astype(int)

        # pick numeric features from df_all to merge for home and away
        numeric_team_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        # remove columns that are clearly not features (if present)
        for junk in ("index",):
            if junk in numeric_team_cols:
                numeric_team_cols.remove(junk)
        # ensure Teams index to merge
        team_feats = df_all.set_index("Teams")[numeric_team_cols].copy()

        # merge home and away features
        merged = hist_parsed.merge(team_feats.add_prefix("home_"), left_on="home_team", right_index=True, how="left")
        merged = merged.merge(team_feats.add_prefix("away_"), left_on="away_team", right_index=True, how="left")

        # drop rows with missing features or target
        feat_cols = [c for c in merged.columns if c.startswith("home_") or c.startswith("away_")]
        merged = merged.dropna(subset=feat_cols + ["home_win"]).copy()

        if merged.shape[0] < 40:
            train_warning = "Not enough complete historical rows after merge to train ML (need >=40). Using baseline."
        else:
            # Build model pipeline
            X = merged[feat_cols]
            y = merged["home_win"].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            pipeline = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=200, random_state=0))])
            pipeline.fit(X_train, y_train)
            model = pipeline
            feature_cols = feat_cols
            st.success(f"Trained ML model on {len(X_train)} rows (test {len(X_test)} rows).")

if train_warning:
    st.warning(train_warning)

# -----------------------
# Load or build schedule
# -----------------------
if schedule_df is None:
    st.info("No Randomized_Schedule.csv found — I'll build a simple randomized schedule (best-effort).")
    # Simple fallback: each team plays 20 conf + 8 nonconf within rules — basic implementation (not full constraint solver)
    teams = df_all["Teams"].dropna().unique().tolist()
    conf_map = df_all.set_index("Teams")["Conference"].to_dict()
    rows = []
    rng = np.random.default_rng(42)
    for team in teams:
        team_conf = conf_map.get(team)
        conf_pool = [t for t in teams if t != team and conf_map.get(t) == team_conf]
        nonconf_pool = [t for t in teams if t != team and conf_map.get(t) != team_conf]
        # sample opponents
        conf_sample = list(rng.choice(conf_pool, size=min(20, max(0, len(conf_pool))), replace=(len(conf_pool)<20))) if conf_pool else []
        nonconf_count = int(rng.integers(8,13))
        nonconf_sample = list(rng.choice(nonconf_pool, size=min(nonconf_count, max(0, len(nonconf_pool))), replace=(len(nonconf_pool)<nonconf_count))) if nonconf_pool else []
        opponents = conf_sample + nonconf_sample
        for opp in opponents:
            day = int(rng.integers(1, 161))
            home = rng.choice([True, False])
            home_team = team if home else opp
            away_team = opp if home else team
            rows.append({"Day": day, "Home": home_team, "Away": away_team, "Conference_Game": (conf_map.get(team)==conf_map.get(opp))})
    schedule_df = pd.DataFrame(rows)
    # basic dedupe & sort
    schedule_df = schedule_df.drop_duplicates(subset=["Day","Home","Away"]).sort_values("Day").reset_index(drop=True)
else:
    # make sure schedule_df columns match expected names
    if "Home" not in schedule_df.columns or "Away" not in schedule_df.columns:
        st.error("Schedule file must contain 'Home' and 'Away' columns.")
        st.stop()
    # ensure Day integer
    if "Day" in schedule_df.columns:
        schedule_df["Day"] = pd.to_numeric(schedule_df["Day"], errors="coerce").fillna(-1).astype(int)
    else:
        schedule_df["Day"] = -1

# -----------------------
# Prediction helpers
# -----------------------
def predict_game_prob(home, away):
    """
    Returns probability that home team wins (0..1) and predicted winner name.
    Uses trained model if available; otherwise uses Average Ranking numeric baseline.
    """
    # If we have trained model, create feature row (home_... and away_...)
    if model is not None and feature_cols is not None:
        # prepare numeric team features from df_all
        numeric_team_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
        # Build home/away series
        try:
            home_series = df_all.set_index("Teams").loc[home, numeric_team_cols].add_prefix("home_")
            away_series = df_all.set_index("Teams").loc[away, numeric_team_cols].add_prefix("away_")
        except KeyError:
            # missing team in All_stats -> fallback
            return 0.5, "Unknown"
        Xrow = pd.concat([home_series, away_series]).reindex(feature_cols).fillna(0).values.reshape(1, -1)
        prob = float(model.predict_proba(Xrow)[0,1])
        pred = home if prob >= 0.5 else away
        return prob, pred

    # fallback baseline: use Average Ranking (lower is better)
    if "Average Ranking" in df_all.columns:
        try:
            home_rank = float(df_all.loc[df_all["Teams"] == home, "Average Ranking"].values[0])
            away_rank = float(df_all.loc[df_all["Teams"] == away, "Average Ranking"].values[0])
            # convert rank diff to probability — smaller rank better => more win prob
            diff = (away_rank - home_rank)  # positive means home is better
            prob = 1 / (1 + np.exp(-diff / 50.0))  # sigmoid scaling
            pred = home if prob >= 0.5 else away
            return float(prob), pred
        except Exception:
            return 0.5, home
    return 0.5, home

# -----------------------
# Predict schedule
# -----------------------
@st.cache_data
def predict_entire_schedule(schedule_df):
    out_rows = []
    for _, r in schedule_df.iterrows():
        h = r["Home"]
        a = r["Away"]
        prob, pred = predict_game_prob(h, a)
        out_rows.append({
            "Day": int(r.get("Day", -1)),
            "Home": h,
            "Away": a,
            "Conference_Game": bool(r.get("Conference_Game", False)),
            "Prob_Home_Win": prob,
            "Pred_Winner": pred
        })
    return pd.DataFrame(out_rows)

pred_df = predict_entire_schedule(schedule_df)

# -----------------------
# UI: selectors
# -----------------------
st.title("Schedule Predictor — View & Download Predictions")
st.markdown("Select a view mode and filter to see predicted outcomes for the randomized schedule.")

st.sidebar.header("View options")
view_by = st.sidebar.selectbox("View by", ["Day", "Team", "Conference"], index=0)

if view_by == "Day":
    min_day = int(pred_df["Day"].min())
    max_day = int(pred_df["Day"].max())
    day_sel = st.sidebar.slider("Select Day", min_value=min_day, max_value=max_day, value=min_day)
    view_df = pred_df[pred_df["Day"] == day_sel].copy()
elif view_by == "Team":
    teams = sorted(pd.unique(np.concatenate([pred_df["Home"].unique(), pred_df["Away"].unique()])))
    team_sel = st.sidebar.selectbox("Select Team", teams)
    view_df = pred_df[(pred_df["Home"] == team_sel) | (pred_df["Away"] == team_sel)].copy()
else:  # Conference
    if "Conference" in df_all.columns:
        confs = sorted(df_all["Conference"].dropna().unique().tolist())
    else:
        confs = ["Unknown"]
    conf_sel = st.sidebar.selectbox("Select Conference", confs)
    teams_in_conf = df_all[df_all["Conference"] == conf_sel]["Teams"].unique().tolist()
    view_df = pred_df[(pred_df["Home"].isin(teams_in_conf)) | (pred_df["Away"].isin(teams_in_conf))].copy()

st.header("Predicted Games")
st.write(f"Showing {len(view_df)} games for filter: {view_by}")

if view_df.empty:
    st.info("No games for this filter.")
else:
    # sort
    view_df = view_df.sort_values(["Day", "Prob_Home_Win"], ascending=[True, False]).reset_index(drop=True)
    view_df["Prob_Home_Win_%"] = (view_df["Prob_Home_Win"] * 100).round(1).astype(str) + "%"
    display_cols = ["Day", "Home", "Away", "Prob_Home_Win_%", "Pred_Winner", "Conference_Game"]
    st.dataframe(view_df[display_cols], use_container_width=True)

    # histogram
    st.subheader("Probability distribution (home win)")
    import plotly.express as px
    fig = px.histogram(view_df, x="Prob_Home_Win", nbins=20, title="Distribution of Home Win Probabilities")
    st.plotly_chart(fig, use_container_width=True)

    # aggregated summary (predicted wins by team)
    st.subheader("Predicted wins (home-favored probabilities summed)")
    agg = view_df.copy()
    agg["Home_Prob_HomeWin"] = agg["Prob_Home_Win"]
    # each match gives fractional credit to predicted winner; show expected wins per team (home-side expected)
    expected_home_wins = agg.groupby("Home")["Home_Prob_HomeWin"].sum().rename("Expected_Home_Wins")
    expected_away_wins = agg.assign(Away_Prob=lambda d: 1 - d["Prob_Home_Win"]).groupby("Away")["Away_Prob"].sum().rename("Expected_Away_Wins")
    expected = pd.concat([expected_home_wins, expected_away_wins], axis=1).fillna(0)
    expected["Expected_Total_Wins"] = expected["Expected_Home_Wins"] + expected["Expected_Away_Wins"]
    expected = expected.sort_values("Expected_Total_Wins", ascending=False).reset_index().rename(columns={"index":"Team"})
    st.dataframe(expected.head(30), use_container_width=True)

    # download filtered view
    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download this view as CSV", data=csv_bytes, file_name="predicted_games_view.csv", mime="text/csv")

# full schedule download
st.markdown("---")
full_csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download full predicted schedule (CSV)", data=full_csv, file_name="predicted_full_schedule.csv", mime="text/csv")

# show training note
if train_warning:
    st.info("Note: ML predictor was not used: " + train_warning + " Baseline ranking used instead.")




