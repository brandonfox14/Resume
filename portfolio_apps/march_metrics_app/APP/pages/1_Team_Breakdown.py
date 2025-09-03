import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("Data/All_stats.csv", encoding="latin1")

df = load_data()

# -----------------------
# Helpers / formatting
# -----------------------
def format_value(key_or_label, val):
    """Format numeric or percent values for display."""
    if pd.isna(val):
        return "N/A"
    try:
        v = float(val)
    except Exception:
        return str(val)
    # treat as percent if key or label indicates percent
    if ("PERC" in str(key_or_label).upper()) or ("%" in str(key_or_label)):
        return f"{v:.1%}" if v <= 1 else f"{v:.1f}%"
    # format integers without .0 if safe
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.1f}"

def format_rank(val):
    """
    If rank is missing (NaN) -> show the requested note.
    If rank exists -> show integer (no .0).
    """
    if val is None:
        return "No rank mapping defined"
    if pd.isna(val) or val == "N/A":
        return "Not enough games played for ranking"
    try:
        return int(float(val))
    except Exception:
        return val

# -----------------------
# Explicit rank mapping (source-of-truth)
# Add or correct entries here to match your CSV EXACTLY.
# -----------------------
rank_overrides = {
    # offense (must match your offense_cols keys exactly)
    "Points": "Points_Rank",
    "FG_PERC": "FG_PERC_Rank",
    "FGM/G": "FGM/G_Rank",
    "FG3_PERC": "FG3_PERC_Rank",
    "FG3M/G": "FG3M/G_Rank",
    "FT_PERC": "FT_PERC_Rank",
    "FTM/G": "FTM/G_Rank",
    "% of Points from 3": "% of Points from 3_Rank",
    "% of shots taken from 3": "% of shots taken from 3_Rank",

    # rebounds / misc
    "OReb": "OReb Rank",
    "OReb chances": "OReb chances Rank",
    "DReb": "DReb Rank",
    "Rebounds": "Rebounds Rank",
    "Rebound Rate": "Rebound Rate Rank",
    "AST": "AST Rank",
    "AST/FGM": "AST/FGM Rank",
    "TO": "TO Rank",
    "STL": "STL Rank",

    # extras
    "PF": "PF_Rank",
    "Foul Differential": "Foul Differential Rank",
    "Extra Scoring Chances": "Extra Scoring Chances Rank",
    "PTS_OFF_TURN": "PTS_OFF_TURN_RANK",
    "FST_BREAK": "FST_BREAK_RANK",
    "PTS_PAINT": "PTS_PAINT_RANK",

    # defense
    "OPP_PPG": "OPP_PPG_RANK",
    "OPP_FG_PERC": "OPP_FG_PERC_Rank",
    "OPP_FGM/G": "OPP_FGM/G_Rank",
    "OPP_FG3_PERC": "OPP_FG3_PERC_Rank",
    "OPP_FG3M/G": "OPP_FG3M/G_Rank",
    "OPP_% of Points from 3": "OPP_% of Points from 3 rank",
    "OPP_% of shots taken from 3": "OPP_% of shots taken from 3 Rank",
    "OPP_OReb": "OPP_OReb_RANK",
}

def get_rank_col(key: str):
    """Return explicit mapping. If missing, return None (no fallback)."""
    return rank_overrides.get(key)

def robust_normalize(df_section: pd.DataFrame) -> pd.DataFrame:
    """Normalize each column to [0,1] with robust handling of constant or empty columns."""
    out = pd.DataFrame(index=df_section.index, columns=df_section.columns, dtype=float)
    for c in df_section.columns:
        col = pd.to_numeric(df_section[c], errors='coerce')
        if col.dropna().empty:
            out[c] = 0.5  # neutral if no data
            continue
        mn = col.min(skipna=True)
        mx = col.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx):
            out[c] = 0.5
        elif mx == mn:
            out[c] = 0.5
        else:
            out[c] = (col - mn) / (mx - mn)
    return out

# -----------------------
# Default team selection: choose team with most wins (robust string match)
# -----------------------
def _norm_name(s):
    if s is None:
        return ""
    return str(s).strip().lower()

teams_sorted = sorted(df["Teams"].dropna().unique().tolist())

default_index = 0
if "Wins" in df.columns:
    # numeric wins if possible
    wins = pd.to_numeric(df["Wins"], errors="coerce")
    if wins.notna().any():
        try:
            idxmax = wins.idxmax()
            default_team_raw = df.at[idxmax, "Teams"]
            # find index in sorted list using normalized compare
            match_idx = next((i for i, t in enumerate(teams_sorted) if _norm_name(t) == _norm_name(default_team_raw)), None)
            if match_idx is not None:
                default_index = match_idx
        except Exception:
            default_index = 0

# -----------------------
# TEAM DROPDOWN
# -----------------------
selected_team = st.selectbox("Select a Team", teams_sorted, index=default_index)
team_data = df[df["Teams"] == selected_team].iloc[0]
team_conf = team_data.get("Conference", None)

# -----------------------
# Section builder
# -----------------------
def build_section_chart(section_cols: dict, section_title: str):
    """
    section_cols: dict mapping CSV key -> pretty label.
    This function uses ONLY get_rank_col(key) to find rank columns.
    """
    st.header(f"{selected_team} {section_title}")

    # check for missing stat columns (fail fast with a clear message)
    missing = [k for k in section_cols.keys() if k not in df.columns]
    if missing:
        st.error(f"Missing columns in dataset required for '{section_title}': {missing}")
        return

    # display stat rows
    for key, label in section_cols.items():
        col1, col2, col3 = st.columns([3, 2, 3])
        with col1:
            st.markdown(f"**{label}**")
        with col2:
            st.write(format_value(key, team_data.get(key, float("nan"))))
        with col3:
            rank_col = get_rank_col(key)
            if rank_col is None:
                st.write("No rank mapping defined")
            else:
                st.write(format_rank(team_data.get(rank_col, pd.NA)))

    # prepare normalized lines
    stat_keys = list(section_cols.keys())
    section_df = df[stat_keys].apply(pd.to_numeric, errors="coerce")
    normalized = robust_normalize(section_df)

    # team normalized row
    team_norm_row = normalized.loc[df["Teams"] == selected_team]
    if team_norm_row.shape[0] == 0:
        team_norm = [0.5] * len(stat_keys)
    else:
        team_norm = team_norm_row.iloc[0].tolist()

    # conference normalized
    conf_norm = None
    if team_conf:
        conf_rows = normalized[df["Conference"] == team_conf]
        if conf_rows.shape[0] > 0:
            conf_norm = conf_rows.mean(skipna=True).tolist()

    league_norm = normalized.mean(skipna=True).tolist()

    # hover texts
    hover_texts = []
    for key, label in section_cols.items():
        val = team_data.get(key, float("nan"))
        rank_col = get_rank_col(key)
        rank_val = format_rank(team_data.get(rank_col, pd.NA)) if rank_col else "No rank mapping defined"
        col_min = section_df[key].min(skipna=True)
        col_max = section_df[key].max(skipna=True)
        conf_avg = df[df["Conference"] == team_conf][key].mean() if team_conf else float("nan")
        league_avg = section_df[key].mean()
        hover_texts.append(
            f"<b>{label}</b><br>"
            f"{selected_team}: {format_value(key, val)} (Rank: {rank_val})<br>"
            f"Min: {format_value(key, col_min)} — Max: {format_value(key, col_max)}<br>"
            f"{team_conf + ' Avg' if team_conf else 'Conf Avg'}: {format_value(key, conf_avg)}<br>"
            f"League Avg: {format_value(key, league_avg)}"
        )

    # plotly line chart (normalized)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(section_cols.values()), y=team_norm, mode="lines+markers",
                             name=selected_team, hoverinfo="text", hovertext=hover_texts))
    if conf_norm is not None:
        fig.add_trace(go.Scatter(x=list(section_cols.values()), y=conf_norm, mode="lines+markers",
                                 name=f"{team_conf} Avg", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=list(section_cols.values()), y=league_norm, mode="lines+markers",
                             name="League Avg", line=dict(dash="dot")))

    fig.update_layout(title=f"{section_title} Comparison (Normalized)",
                      yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
                      xaxis=dict(tickangle=45),
                      plot_bgcolor="white",
                      margin=dict(t=60, b=120))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Define the four sections
# -------------------------------
offense_cols = {
    "Points": "Points Per Game",
    "FG_PERC": "Field Goal Percentage",
    "FGM/G": "Field Goals Made per Game",
    "FG3_PERC": "3 Point Field Goal Percentage",
    "FG3M/G": "3 Point Field Goals Made per Game",
    "FT_PERC": "Free Throw Percentage",
    "FTM/G": "Free Throws Made per Game",
    "% of Points from 3": "Percent of Points from 3",
    "% of shots taken from 3": "Percent of Shots Taken from 3"
}

defense_cols = {
    "OPP_PPG": "Opponent Points Per Game",
    "OPP_FG_PERC": "Opponent Field Goal Percentage",
    "OPP_FGM/G": "Opponent FGM per Game",
    "OPP_FG3_PERC": "Opponent 3PT Percentage",
    "OPP_FG3M/G": "Opponent 3PTM per Game",
    "OPP_% of Points from 3": "Opponent % of Points from 3",
    "OPP_% of shots taken from 3": "Opponent % of Shots Taken from 3",
    "OPP_OReb": "Opponent Offensive Rebounds"
}

other_cols = {
    "OReb": "Offensive Rebounds",
    "OReb chances": "Offensive Rebound Rate",
    "DReb": "Defensive Rebounds",
    "Rebounds": "Total Rebounds",
    "Rebound Rate": "Rebound Rate",
    "AST": "Assists",
    "AST/FGM": "Assists per Field Goal Made",
    "TO": "Turnovers",
    "STL": "Steals"
}

extras_cols = {
    "Extra Scoring Chances": "Extra Scoring Chances",
    "PTS_OFF_TURN": "Points Off Turnovers",
    "FST_BREAK": "Fast Break Points",
    "PTS_PAINT": "Points in Paint",
    "PF": "Personal Fouls",
    "Foul Differential": "Foul Differential"
}

# -------------------------------
# Build charts
# -------------------------------
build_section_chart(offense_cols, "Offensive Statistics")
build_section_chart(defense_cols, "Defensive Statistics")
build_section_chart(other_cols, "Rebounds / AST / TO / STL")
build_section_chart(extras_cols, "Extra Statistics")



# I need to adjust some of the groupings, add steal to turnover ratio, 
# and add headers to the offensive columns to explain that rankings is the far right column
