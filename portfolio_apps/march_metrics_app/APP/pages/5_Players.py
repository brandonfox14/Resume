import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --------------------
# Load Data
# --------------------
@st.cache_data
def load_data():
    return pd.read_csv("Data/All_stats.csv", encoding="latin1")

df = load_data()

# --------------------
# Default team = max "Championship Criteria"
# --------------------
teams_sorted = sorted(df["Teams"].dropna().unique().tolist())
if "Championship Criteria" in df.columns:
    champ_vals = pd.to_numeric(df["Championship Criteria"], errors="coerce")
    if champ_vals.notna().any():
        default_team = df.loc[champ_vals.idxmax(), "Teams"]
    else:
        default_team = teams_sorted[0]
else:
    default_team = teams_sorted[0]

team_choice = st.selectbox(
    "Select Team",
    teams_sorted,
    index=max(0, teams_sorted.index(default_team)) if default_team in teams_sorted else 0
)

conf = df.loc[df["Teams"] == team_choice, "Conference"].values[0] if "Conference" in df.columns else "Conference"

team_df = df[df["Teams"] == team_choice].copy()
conf_df = df[df["Conference"] == conf].copy() if "Conference" in df.columns else df.copy()

# --------------------
# Drop unnecessary columns
# --------------------
drop_cols = [
    "FGM_TOP7", "FGA-Top7", "FG3sM-Top7", "FG3sA-Top7", "FTM-Top7", "FTA-Top7",
    "FGA-Top7-Perc", "FG3sA-Top7-Perc", "FTA-Top7-Perc"
]
team_df.drop(columns=[c for c in drop_cols if c in team_df.columns], inplace=True, errors="ignore")
conf_df.drop(columns=[c for c in drop_cols if c in conf_df.columns], inplace=True, errors="ignore")

# --------------------
# Convert fractions -> percentages
# --------------------
percent_cols = [
    "FG_PERC-Top7", "FG3_PERC-Top7", "FT_PERC-Top7",
    "FG_PERC_Top7_per", "FG3_PERC_Top7_per", "FT_PERC_Top7_per",
    "OReb-Top7-Perc", "DReb-Top7-Perc", "Rebounds-Top7-Perc",
    "AST-Top7-Perc", "TO-Top7-Perc", "STL-Top7-Perc", "Points-Top7-Perc",
    "Start Percentage top 7",
    "FGM-Top7-Perc", "FG3sM-Top7-Perc", "FTM-Top7-Perc",
]
for col in percent_cols:
    if col in team_df.columns:
        team_df[col] = pd.to_numeric(team_df[col], errors="coerce") * 100
    if col in conf_df.columns:
        conf_df[col] = pd.to_numeric(conf_df[col], errors="coerce") * 100

numeric_cols_extra = [
    "OReb-Top7", "DReb-Top7", "Rebounds-Top7", "AST-Top7",
    "TO-Top7", "STL-Top7", "Points per Game-Top7"
]
for col in numeric_cols_extra:
    if col in team_df.columns:
        team_df[col] = pd.to_numeric(team_df[col], errors="coerce")
    if col in conf_df.columns:
        conf_df[col] = pd.to_numeric(conf_df[col], errors="coerce")

# --------------------
# Rename columns
# --------------------
rename_core = {
    "FG_PERC-Top7": "Field Goal Percentage",
    "FG3_PERC-Top7": "3 Field Goal Percentage",
    "FT_PERC-Top7": "Free Throw Percentage",
    "OReb-Top7": "Offensive Rebounds Per Game",
    "DReb-Top7": "Defensive Rebounds Per Game",
    "Rebounds-Top7": "Rebounds Per Game",
    "AST-Top7": "Assists Per Game",
    "TO-Top7": "Turnover Per Game",
    "STL-Top7": "Steals Per Game",
    "Points per Game-Top7": "Points Per Game",
    "Start Percentage top 7": "Starting Percentage",
}

rename_pct_of_team = {
    "FG_PERC_Top7_per": "Core 7 Percentage of Team Field Goal Percentage",
    "FG3_PERC_Top7_per": "Core 7 Percentage of Team 3 Point Field Goal Percentage",
    "FT_PERC_Top7_per": "Core 7 Percentage of Team Free Throw Percentage",
    "OReb-Top7-Perc": "Core 7 Percentage of Team Offensive Rebounds",
    "DReb-Top7-Perc": "Core 7 Percentage of Team Defensive Rebounds",
    "Rebounds-Top7-Perc": "Core 7 Percentage of Team Rebounds",
    "AST-Top7-Perc": "Core 7 Percentage of Team Assistants",
    "TO-Top7-Perc": "Core 7 Percentage of Team Turnovers",
    "STL-Top7-Perc": "Core 7 Percentage of Team Steals",
    "Points-Top7-Perc": "Core 7 Percentage of Team Points",
    "FGM-Top7-Perc": "Core 7 Percentage of Team Field Goals Made",
    "FG3sM-Top7-Perc": "Core 7 Percentage of Team 3 Field Goals Made",
    "FTM-Top7-Perc": "Core 7 Percentage of Team Free Throws Made",
}

team_df_core = team_df.rename(columns=rename_core)
conf_df_core = conf_df.rename(columns=rename_core)

team_df_pct = team_df.rename(columns=rename_pct_of_team)
conf_df_pct = conf_df.rename(columns=rename_pct_of_team)

# --------------------
# Build Summary Tables
# --------------------
core_cols_labels = [
    "Field Goal Percentage", "3 Field Goal Percentage", "Free Throw Percentage",
    "Offensive Rebounds Per Game", "Defensive Rebounds Per Game", "Rebounds Per Game",
    "Assists Per Game", "Turnover Per Game", "Steals Per Game", "Points Per Game",
    "Starting Percentage",
]
summary_core = pd.DataFrame({
    "Stat": core_cols_labels,
    "Team Value": [team_df_core.iloc[0].get(c, np.nan) for c in core_cols_labels],
    "Conference Average": [conf_df_core[c].mean() if c in conf_df_core.columns else np.nan for c in core_cols_labels],
})

pct_team_cols_labels = [
    "Core 7 Percentage of Team Field Goal Percentage",
    "Core 7 Percentage of Team 3 Point Field Goal Percentage",
    "Core 7 Percentage of Team Free Throw Percentage",
    "Core 7 Percentage of Team Offensive Rebounds",
    "Core 7 Percentage of Team Defensive Rebounds",
    "Core 7 Percentage of Team Rebounds",
    "Core 7 Percentage of Team Assistants",
    "Core 7 Percentage of Team Turnovers",
    "Core 7 Percentage of Team Steals",
    "Core 7 Percentage of Team Points",
    "Core 7 Percentage of Team Field Goals Made",
    "Core 7 Percentage of Team 3 Field Goals Made",
    "Core 7 Percentage of Team Free Throws Made",
]
summary_stats = pd.DataFrame({
    "Stat": pct_team_cols_labels,
    "Team Value": [team_df_pct.iloc[0].get(c, np.nan) for c in pct_team_cols_labels],
    "Conference Average": [conf_df_pct[c].mean() if c in conf_df_pct.columns else np.nan for c in pct_team_cols_labels],
})

for df_ in (summary_core, summary_stats):
    df_["Team Value"] = pd.to_numeric(df_["Team Value"], errors="coerce")
    df_["Conference Average"] = pd.to_numeric(df_["Conference Average"], errors="coerce")

# --------------------
# Show Summary Tables
# --------------------
st.subheader(f"{team_choice} Core 7 Players Statistics")
st.dataframe(summary_core, use_container_width=True)

st.subheader(f"{team_choice} Percent of Team Stats for Core 7 Players")
st.dataframe(summary_stats, use_container_width=True)

# --------------------
# Visual 1a: Percentages Chart
# --------------------
percent_cols_chart = [
    "Field Goal Percentage", "3 Field Goal Percentage", "Free Throw Percentage",
    "Points Per Game", "Starting Percentage"
]
summary_percent = summary_core[summary_core["Stat"].isin(percent_cols_chart)]

fig1a = px.bar(
    summary_percent,
    x="Stat",
    y=["Team Value", "Conference Average"],
    barmode="group",
    title=f"{team_choice} vs {conf} – Core 7 Percentages & Points"
)
st.plotly_chart(fig1a, use_container_width=True)

# --------------------
# Visual 1b: Counting Stats Chart
# --------------------
counting_cols_chart = [
    "Offensive Rebounds Per Game", "Defensive Rebounds Per Game", "Rebounds Per Game",
    "Assists Per Game", "Turnover Per Game", "Steals Per Game"
]
summary_counting = summary_core[summary_core["Stat"].isin(counting_cols_chart)]

fig1b = px.bar(
    summary_counting,
    x="Stat",
    y=["Team Value", "Conference Average"],
    barmode="group",
    title=f"{team_choice} vs {conf} – Core 7 Counting Stats"
)
st.plotly_chart(fig1b, use_container_width=True)

# --------------------
# Visual 2: Percent-of-team bars (still with ranking overlay)
# --------------------
label_to_orig = {v: k for k, v in rename_pct_of_team.items()}
orig_cols_for_rank = [label_to_orig[lbl] for lbl in pct_team_cols_labels if lbl in label_to_orig]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=summary_stats["Stat"],
    y=summary_stats["Team Value"],
    name=f"{team_choice} Value"
))
fig2.add_trace(go.Bar(
    x=summary_stats["Stat"],
    y=summary_stats["Conference Average"],
    name=f"{conf} Avg"
))
fig2.update_layout(
    title=f"{team_choice} Percent of Team Stats for Core 7 Players",
    yaxis=dict(title="Percent / Value")
)
st.plotly_chart(fig2, use_container_width=True)
