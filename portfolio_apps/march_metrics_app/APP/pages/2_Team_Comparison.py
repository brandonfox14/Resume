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
# Explicit rank mapping (use for all stats)
# -----------------------
rank_overrides = {
    "Points": "Points_Rank",
    "FG_PERC": "FG_PERC_Rank",
    "FGM/G": "FGM/G_Rank",
    "FG3_PERC": "FG3_PERC_Rank",
    "FG3M/G": "FG3M/G_Rank",
    "FT_PERC": "FT_PERC_Rank",
    "FTM/G": "FTM/G_Rank",
    "% of Points from 3": "% of Points from 3_Rank",
    "% of shots taken from 3": "% of shots taken from 3_Rank",
    "OReb": "OReb Rank",
    "OReb chances": "OReb chances Rank",
    "DReb": "DReb Rank",
    "Rebounds": "Rebounds Rank",
    "Rebound Rate": "Rebound Rate Rank",
    "AST": "AST Rank",
    "AST/FGM": "AST/FGM Rank",
    "TO": "TO Rank",
    "STL": "STL Rank",
    "PF": "PF_Rank",
    "Foul Differential": "Foul Differential Rank",
    "Extra Scoring Chances": "Extra Scoring Chances Rank",
    "PTS_OFF_TURN": "PTS_OFF_TURN_RANK",
    "FST_BREAK": "FST_BREAK_RANK",
    "PTS_PAINT": "PTS_PAINT_RANK",
    "OPP_PPG": "OPP_PPG_RANK",
    "OPP_FG_PERC": "OPP_FG_PERC_Rank",
    "OPP_FGM/G": "OPP_FGM/G_Rank",
    "OPP_FG3_PERC": "OPP_FG3_PERC_Rank",
    "OPP_FG3M/G": "OPP_FG3M/G_Rank",
    "OPP_% of Points from 3": "OPP_% of Points from 3 rank",
    "OPP_% of shots taken from 3": "OPP_% of shots taken from 3 Rank",
    "OPP_OReb": "OPP_OReb_RANK",
}

# -----------------------
# Define stat groups
# -----------------------
stat_groups = {
    "Offense": ["Points","FG_PERC","FGM/G","FG3_PERC","FG3M/G","% of Points from 3",
                "% of shots taken from 3","FT_PERC","FTM/G"],
    "Defense": ["OPP_PPG","OPP_FG_PERC","OPP_FGM/G","OPP_FG3_PERC","OPP_FG3M/G",
                "OPP_% of Points from 3","OPP_% of shots taken from 3","OPP_OReb"],
    "Rebounds/AST/TO/STL": ["OReb","OReb chances","DReb","Rebounds","Rebound Rate",
                             "AST","AST/FGM","TO","STL"],
    "Extras": ["Extra Scoring Chances","PTS_OFF_TURN","FST_BREAK","PTS_PAINT","PF","Foul Differential"]
}

# -----------------------
# Team Selection
# -----------------------
teams_sorted = sorted(df["Teams"].dropna().unique().tolist())
col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Left Team", teams_sorted, index=0)
with col2:
    team_b = st.selectbox("Select Right Team", teams_sorted, index=1)

team_a_data = df[df["Teams"]==team_a].iloc[0]
team_b_data = df[df["Teams"]==team_b].iloc[0]

# -----------------------
# Pop-up for missing ranks
# -----------------------
missing_ranks_a = [stat for group in stat_groups.values() for stat in group
                   if pd.isna(team_a_data.get(rank_overrides[stat], np.nan))]
missing_ranks_b = [stat for group in stat_groups.values() for stat in group
                   if pd.isna(team_b_data.get(rank_overrides[stat], np.nan))]

if missing_ranks_a:
    st.warning(f"⚠️ {team_a} has not played enough games for rankings in: {', '.join(missing_ranks_a)}")
if missing_ranks_b:
    st.warning(f"⚠️ {team_b} has not played enough games for rankings in: {', '.join(missing_ranks_b)}")

# -----------------------
# Helper Functions
# -----------------------
def color_by_rank(rank):
    if pd.isna(rank):
        return "lightgrey"
    rank = int(rank)
    if rank > 200:
        return f"rgba(255,100,100,0.8)"  # lighter red
    elif 151 <= rank <= 200:
        return f"rgba(180,180,180,0.7)"  # grey
    else:
        green_val = int(50 + (150 - rank) * 1.4)
        return f"rgba(0,{green_val},0,0.8)"

def normalize_stat(val, stat_col):
    min_val = df[stat_col].min()
    max_val = df[stat_col].max()
    if pd.isna(val) or min_val==max_val:
        return 0.5
    return (val - min_val)/(max_val - min_val)

# -----------------------
# Side-by-Side Bars
# -----------------------
st.subheader("Team Comparison: Stats")
for group_name, stats in stat_groups.items():
    st.markdown(f"### {group_name}")
    for stat in stats:
        rank_col = rank_overrides[stat]
        val_a = team_a_data.get(stat,np.nan)
        val_b = team_b_data.get(stat,np.nan)
        rank_a = team_a_data.get(rank_col,np.nan)
        rank_b = team_b_data.get(rank_col,np.nan)
        norm_a = normalize_stat(val_a, stat)
        norm_b = normalize_stat(val_b, stat)
        color_a = color_by_rank(rank_a)
        color_b = color_by_rank(rank_b)
        
        col_left, col_center, col_right = st.columns([4,2,4])
        with col_left:
            st.markdown(f"<div style='text-align:right; background-color:{color_a}; width:{int(norm_a*100)}%; padding:5px; border-radius:5px;'>{val_a}</div>", unsafe_allow_html=True)
        with col_center:
            st.markdown(f"**{stat}**", unsafe_allow_html=True)
        with col_right:
            st.markdown(f"<div style='text-align:left; background-color:{color_b}; width:{int(norm_b*100)}%; padding:5px; border-radius:5px;'>{val_b}</div>", unsafe_allow_html=True)

# -----------------------
# Radar Chart (Simplified, inverted axis)
# -----------------------
st.subheader("Team Radar: Average Rankings")
radar_categories = ["Overall","Offense","Defense","Rebounds/AST/TO/STL","Extras"]

def avg_rank(team_data, stats_list):
    # Just take the actual rank values directly
    ranks = [team_data[s] for s in stats_list if not pd.isna(team_data[s])]
    return np.mean(ranks) if ranks else np.nan

# Compute average rankings for each category
overall_a = team_a_data["Average Ranking"]
overall_b = team_b_data["Average Ranking"]

offense_a = avg_rank(team_a_data, stat_groups["Offense"])
defense_a = avg_rank(team_a_data, stat_groups["Defense"])
reb_a = avg_rank(team_a_data, stat_groups["Rebounds/AST/TO/STL"])
extras_a = avg_rank(team_a_data, stat_groups["Extras"])

offense_b = avg_rank(team_b_data, stat_groups["Offense"])
defense_b = avg_rank(team_b_data, stat_groups["Defense"])
reb_b = avg_rank(team_b_data, stat_groups["Rebounds/AST/TO/STL"])
extras_b = avg_rank(team_b_data, stat_groups["Extras"])

# Build radar chart
fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=[overall_a, offense_a, defense_a, reb_a, extras_a],
    theta=radar_categories,
    fill='toself',
    name=team_a
))
fig.add_trace(go.Scatterpolar(
    r=[overall_b, offense_b, defense_b, reb_b, extras_b],
    theta=radar_categories,
    fill='toself',
    name=team_b
))

# Invert radial axis: best (1) on outside, worst (365) on inside
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[365, 1],  # invert axis
            tickvals=[50,100,150,200,250,300,350]
        )
    ),
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)



# need to fix the chart at the bottom I also want to add steal to turnover ratio, and I need to adjust the groupings of stats.
# also fix names, need them to be readable not abreviations. 
