import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("Data/All_stats.csv", encoding="latin1")

df = load_data()

# -----------------------
# Pick default team = highest CLUTCH_FGM
# -----------------------
default_team = df.loc[df["CLUTCH_FGM"].idxmax(), "Teams"]
teams_sorted = sorted(df["Teams"].dropna().unique().tolist())
team_name = st.selectbox("Select Team", teams_sorted, index=teams_sorted.index(default_team), key="clutch_team")

team_data = df[df["Teams"] == team_name].iloc[0]

# -----------------------
# Common name mapping
# -----------------------
stat_name_map = {
    "CLUTCH_FGPERC": "Clutch Field Goal Percentage",
    "CLUTCH_3FGPERC": "Clutch 3 Point Field Goal Percentage",
    "CLUTCH_FTPERC": "Clutch Free Throw Field Goal Percentage",
    "CLUTCH_SM": "Clutch Scoring Margin",
    "CLUTCH_REB": "Average Clutch Rebounds",
    "OPP_CLTCH_REB": "Average Opponent Clutch Rebounds",
    "CLTCH_OFF_REB": "Average Clutch Offensive Rebounds",
    "OPP_CLTCH_OFF_REB": "Average Clutch Opponent Offensive Rebounds",
    "CLTCH_TURN": "Average Clutch Turnovers",
    "CLTCH_OPP_TURN": "Average Clutch Opponent Turnovers",
    "CLTCH_STL": "Average Clutch Steals",
    "TOP25_CLUTCH": "Clutch Games Against Top 25 Opponents",
    "OVERTIME_GAMES": "Overtime Games"
}

# -----------------------
# Define stat/rank pairs
# -----------------------
stat_pairs = [
    ("CLUTCH_FGPERC", "CLUTCH_FG_RANK"),
    ("CLUTCH_3FGPERC", "CLUTCH_3_RANK"),
    ("CLUTCH_FTPERC", "CLUTCH_FT_RANK"),
    ("CLUTCH_SM", "CLUTCH_SM_RANK"),
    ("CLUTCH_REB", "CLUTCH_REB_RANK"),
    ("OPP_CLTCH_REB", "OPP_CLTCH_REB_RANK"),
    ("CLTCH_OFF_REB", "CLTCH_OFF_REB_RANK"),
    ("OPP_CLTCH_OFF_REB", "OPP_CLTCH_OFF_REB_RANK"),
    ("CLTCH_TURN", "CLTCH_TURN_RANK"),
    ("CLTCH_OPP_TURN", "CLTCH_OPP_TURN_RANK"),
    ("CLTCH_STL", "CLTCH_STL_RANK"),
]

extra_stats = ["TOP25_CLUTCH", "OVERTIME_GAMES"]

# -----------------------
# Build Summary Table
# -----------------------
st.subheader("Clutch Performance Summary")

summary_rows = []
for stat, rank in stat_pairs:
    summary_rows.append({
        "Stat": stat_name_map.get(stat, stat),
        "Value": team_data.get(stat, np.nan),
        "Rank": team_data.get(rank, np.nan)
    })

# Add extras at the bottom
for stat in extra_stats:
    summary_rows.append({
        "Stat": stat_name_map.get(stat, stat),
        "Value": team_data.get(stat, np.nan),
        "Rank": None
    })

summary_df = pd.DataFrame(summary_rows)

# If no clutch data, show warning
if summary_df["Value"].isna().any():
    st.warning(f"{team_name} has no clutch games.")
else:
    st.dataframe(summary_df, use_container_width=True)

    # -----------------------
    # Visualization: Shooting % Clutch vs Season
    # -----------------------
    st.subheader("Shooting: Clutch vs Season")

    shooting_stats = ["FG%", "3PT%", "FT%"]
    season_cols = ["FG_PERC", "FG3_PERC", "FT_PERC"]
    clutch_cols = ["CLUTCH_FGPERC", "CLUTCH_3FGPERC", "CLUTCH_FTPERC"]

    season_values = [team_data[c] * 100 for c in season_cols]  # season needs *100
    clutch_values = [team_data[c] for c in clutch_cols]        # clutch already in percent

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shooting_stats,
        y=season_values,
        name="Season",
        marker_color="lightblue"
    ))
    fig.add_trace(go.Bar(
        x=shooting_stats,
        y=clutch_values,
        name="Clutch",
        marker_color="orange"
    ))
    fig.update_layout(
        barmode="group",
        title=f"{team_name} Shooting: Season vs Clutch",
        yaxis=dict(title="Percentage"),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
