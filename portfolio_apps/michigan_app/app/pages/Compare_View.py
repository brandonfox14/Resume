"""
compare_app.py

Streamlit app: Lab <-> Survey Comparison tab for EEG (OpenNeuro) vs NHIS (2024)
Assumes:
  eeg-brfss-app/data/clean/eeg_summary.csv
  eeg-brfss-app/data/clean/nhis_sleep_demo_clean.csv

Features:
 - Inspect available columns and auto-detect likely EEG & survey vars
 - Dropdowns to pick lab metric and survey metric
 - Side-by-side visualizations (bar / box / violin)
 - Grouping options for NHIS (demographic) and EEG (condition/eyes/participant)
 - Summary stats and a conceptual correlation (distribution correlation via Spearman)
 - Data preview and CSV download
 - Plain-language captions and assumptions reminder
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# -----------------------
# Configuration / Paths
# -----------------------
EEG_PATH = Path("data/clean/eeg_summary.csv")
NHIS_PATH = Path("data/clean/nhis_sleep_demo_clean.csv")

# -----------------------
# Utilities
# -----------------------
@st.cache_data
def load_csv(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

def suggest_columns(df, keywords):
    """Return columns containing any of the keywords (case-insensitive)."""
    if df is None:
        return []
    cols = df.columns.tolist()
    hits = []
    for k in keywords:
        for c in cols:
            if k.lower() in c.lower() and c not in hits:
                hits.append(c)
    return hits

def numeric_columns(df):
    if df is None:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()

def safe_corr(series_a, series_b, method="spearman"):
    # remove na
    mask = series_a.notna() & series_b.notna()
    if mask.sum() < 3:
        return np.nan, np.nan
    try:
        if method == "pearson":
            r, p = stats.pearsonr(series_a[mask], series_b[mask])
        else:
            r, p = stats.spearmanr(series_a[mask], series_b[mask])
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan

# -----------------------
# Load data
# -----------------------
st.set_page_config(page_title="EEG ↔ NHIS Compare", layout="wide")
st.title("Lab ↔ Survey Comparison — EEG & NHIS (exploratory, non-diagnostic)")

eeg_df = load_csv(EEG_PATH)
nhis_df = load_csv(NHIS_PATH)

if eeg_df is None:
    st.error(f"EEG CSV not found at `{EEG_PATH}`. Update path at top of file if needed.")
if nhis_df is None:
    st.error(f"NHIS CSV not found at `{NHIS_PATH}`. Update path at top of file if needed.")

if eeg_df is None or nhis_df is None:
    st.stop()

# -----------------------
# Quick previews & column suggestions
# -----------------------
with st.expander("Preview datasets & column suggestions (open if you want to verify)"):
    st.subheader("EEG summary (top 5 rows)")
    st.dataframe(eeg_df.head())

    st.subheader("NHIS sleep demo (top 5 rows)")
    st.dataframe(nhis_df.head())

    st.write("Detected numeric columns (EEG):", numeric_columns(eeg_df))
    st.write("Detected numeric columns (NHIS):", numeric_columns(nhis_df))

    st.markdown("---")
    st.write("Auto-suggested EEG columns by keyword (alpha/theta/power/pvt/panas):")
    st.write(suggest_columns(eeg_df, ["alpha", "theta", "beta", "delta", "power", "pvt", "rt", "panas", "mood", "lapse", "vigil"]))
    st.write("Auto-suggested NHIS columns by keyword (sleep/hours/rest/sleepaid/trouble):")
    st.write(suggest_columns(nhis_df, ["sleep", "hour", "rest", "trouble", "aid", "nap", "insomnia"]))

# -----------------------
# Prepare dropdown options (friendly names -> column names)
# -----------------------
# Build EEG metric options
eeg_numeric = numeric_columns(eeg_df)
eeg_suggested = suggest_columns(eeg_df, ["alpha", "theta", "beta", "delta", "power", "pvt", "rt", "panas", "mood"])
# Keep ordering: suggested first then other numeric
ordered_eeg = eeg_suggested + [c for c in eeg_numeric if c not in eeg_suggested]

# Build NHIS metric options
nhis_numeric = numeric_columns(nhis_df)
nhis_suggested = suggest_columns(nhis_df, ["sleep", "hour", "rest", "trouble", "aid", "insomnia"])
ordered_nhis = nhis_suggested + [c for c in nhis_numeric if c not in nhis_suggested]

# Add friendly labels mapping
friendly_labels = {
    "n_channels": "Number of EEG Channels",
    "duration_sec": "Recording Length (seconds)",
    "sampling_rate": "EEG Sampling Rate (Hz)",
    "theta_mean": "Average Theta Brainwave Power",
    "alpha_mean": "Average Alpha Brainwave Power",
    "beta_mean": "Average Beta Brainwave Power",
    "Age": "Participant Age",
    "PVT_item1_NS": "PVT Reaction Time – Item 1 (Normal Sleep)",
    "PVT_item2_NS": "PVT Reaction Time – Item 2 (Normal Sleep)",
    "PVT_item3_NS": "PVT Reaction Time – Item 3 (Normal Sleep)",
    "PVT_item1_SD": "PVT Reaction Time – Item 1 (Sleep Deprived)",
    "PVT_item2_SD": "PVT Reaction Time – Item 2 (Sleep Deprived)",
    "PVT_item3_SD": "PVT Reaction Time – Item 3 (Sleep Deprived)",
    "PANAS_P_NS": "Positive Mood Score (Normal Sleep)",
    "PANAS_P_SD": "Positive Mood Score (Sleep Deprived)",
    "PANAS_N_NS": "Negative Mood Score (Normal Sleep)",
    "PANAS_N_SD": "Negative Mood Score (Sleep Deprived)",
    "ATQ_NS": "Attention Control Score (Normal Sleep)",
    "ATQ_SD": "Attention Control Score (Sleep Deprived)",
    "SAI_NS": "Anxiety Score (Normal Sleep)",
    "SAI_SD": "Anxiety Score (Sleep Deprived)",
    "SSS_NS": "Sleepiness Scale (Normal Sleep)",
    "SSS_SD": "Sleepiness Scale (Sleep Deprived)",
    "KSS_NS": "Karolinska Sleepiness Score (Normal Sleep)",
    "KSS_SD": "Karolinska Sleepiness Score (Sleep Deprived)",
    "SleepDiary_item3_NS": "Sleep Diary – Time Asleep (Normal Sleep)",
    "EQ": "Empathy Score",
    "Buss_Perry": "Aggression Questionnaire Score",
    "PSQI_GlobalScore": "Sleep Quality Score (Global)",
    "PSQI_item1": "Sleep Quality – Component 1",
    "PSQI_item2": "Sleep Quality – Component 2",
    "PSQI_item3": "Sleep Quality – Component 3",
    "PSQI_item4": "Sleep Quality – Component 4",
    "PSQI_item5": "Sleep Quality – Component 5",
    "PSQI_item6": "Sleep Quality – Component 6",
    "PSQI_item7": "Sleep Quality – Component 7",
    "SLPMED3_A": "Sleep Medication Use (CBD)",
    "SLPMED2_A": "Sleep Medication Use (Over the Counter)",
    "SLPMED1_A": "Sleep Medication Use (Doctor Prescribed)",
    "SLPMEDINTRO_A": "Sleep Medication Introduction Question",
    "SLPSTY_A": "Trouble Staying Asleep",
    "SLPFLL_A": "Trouble Falling Asleep",
    "SLPREST_A": "Days Waking Feeling Rested",
    "SLPHOURS_A": "Hours of Sleep in a 24 Hour Period",
    "SEX_A": "Sex",
    "AGEP_A": "Age",
    "EDUCP_A": "Education Level"
}

def get_friendly_label(col):
    return friendly_labels.get(col, col.replace("_", " ").title())

def labelize(col):
    return get_friendly_label(col)

# Build EEG metric options
eeg_numeric = numeric_columns(eeg_df)
eeg_suggested = suggest_columns(eeg_df, ["alpha", "theta", "beta", "delta", "power", "pvt", "rt", "panas", "mood"])
# Keep ordering: suggested first then other numeric
ordered_eeg = eeg_suggested + [c for c in eeg_numeric if c not in eeg_suggested]

# Build NHIS metric options
nhis_numeric = numeric_columns(nhis_df)
nhis_suggested = suggest_columns(nhis_df, ["sleep", "hour", "rest", "trouble", "aid", "insomnia"])
ordered_nhis = nhis_suggested + [c for c in nhis_numeric if c not in nhis_suggested]

# Map friendly names to actual column names for dropdowns
eeg_options = {get_friendly_label(c): c for c in ordered_eeg}
nhis_options = {get_friendly_label(c): c for c in ordered_nhis}

# Sidebar controls
st.sidebar.header("Compare controls")
lab_metric_label = st.sidebar.selectbox("Choose EEG (Lab) metric", list(eeg_options.keys()))
lab_metric = eeg_options[lab_metric_label]

survey_metric_label = st.sidebar.selectbox("Choose NHIS (Survey) metric", list(nhis_options.keys()))
survey_metric = nhis_options[survey_metric_label]

# EEG grouping (condition likely exists)
eeg_group_cols = [c for c in eeg_df.columns if eeg_df[c].nunique() < 30 and eeg_df[c].dtype == object]
# provide common choices
default_eeg_group = None
for guess in ["condition", "Condition", "sleep_condition", "cond", "eyes", "eyes_state"]:
    if guess in eeg_df.columns:
        default_eeg_group = guess
        break
eeg_group_by = st.sidebar.selectbox("Group EEG by (for aggregation/plot)", options=[None] + eeg_group_cols, index=0 if default_eeg_group is None else (1 + eeg_group_cols.index(default_eeg_group)))

# NHIS grouping (demographic)
nhis_group_cols = [c for c in nhis_df.columns if nhis_df[c].nunique() < 200]  # include many demographic possibilities
default_nhis_group = None
for guess in ["age_group", "agecat", "sex", "gender", "education", "race", "race_ethnicity"]:
    if guess in nhis_df.columns:
        default_nhis_group = guess
        break
nhis_group_by = st.sidebar.selectbox("Group NHIS by (demographic)", options=[None] + nhis_group_cols, index=0 if default_nhis_group is None else (1 + nhis_group_cols.index(default_nhis_group)))

# Plot types and aggregation
agg_func = st.sidebar.selectbox("Aggregation for means", ["mean", "median"])
plot_kind = st.sidebar.radio("Plot style for each panel", ["bar", "box", "violin"], index=0)

# Extra options
show_points = st.sidebar.checkbox("Overlay raw points (when applicable)", value=True)
correlation_method = st.sidebar.selectbox("Distribution correlation method", ["spearman", "pearson"])

# -----------------------
# Main layout: two columns side-by-side
# -----------------------
col1, col2 = st.columns([1,1])

# Left: EEG visualization
with col1:
    st.subheader("Lab (EEG) — distribution by group")
    st.markdown(f"**Metric:** `{lab_metric}` — *{lab_metric_label}*")

    # if grouping is provided: aggregate and plot grouped means
    if eeg_group_by:
        # if group col is numeric but few unique, convert to string
        if eeg_df[eeg_group_by].dtype != object:
            eeg_df[eeg_group_by] = eeg_df[eeg_group_by].astype(str)

        group_df = eeg_df.groupby(eeg_group_by)[lab_metric].agg([np.mean, np.std, np.median, "count"]).reset_index()
        group_df = group_df.rename(columns={"mean": "mean", "std": "std", "median": "median", "count": "n"})
        st.write("Group summary:")
        st.dataframe(group_df.sort_values("mean", ascending=False))

        # plotting
        if plot_kind == "bar":
            fig = px.bar(group_df, x=eeg_group_by, y="mean", error_y="std", labels={eeg_group_by: labelize(eeg_group_by), "mean": lab_metric_label})
            st.plotly_chart(fig, use_container_width=True)
        else:
            # box/violin: use raw data grouped
            fig = px.box(eeg_df, x=eeg_group_by, y=lab_metric, points="all" if show_points else "outliers", labels={eeg_group_by: labelize(eeg_group_by), lab_metric: lab_metric_label})
            if plot_kind == "violin":
                fig = px.violin(eeg_df, x=eeg_group_by, y=lab_metric, box=True, points="all" if show_points else "outliers", labels={eeg_group_by: labelize(eeg_group_by), lab_metric: lab_metric_label})
            st.plotly_chart(fig, use_container_width=True)

    else:
        # no grouping: global summary and distribution
        desc = eeg_df[lab_metric].describe()
        st.write("Global summary stats:")
        st.table(desc.to_frame(name=lab_metric_label).T)
        if plot_kind == "bar":
            st.info("Bar requires grouping; using binned distribution instead.")
            # create bins
            binned = pd.cut(eeg_df[lab_metric].dropna(), bins=8)
            binned_counts = eeg_df.groupby(binned)[lab_metric].count().rename("count").reset_index()
            fig = px.bar(binned_counts, x=binned_counts.index.astype(str), y="count", labels={"x":"Value bin", "count":"Count"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            if plot_kind == "box":
                fig = px.box(eeg_df, y=lab_metric, points="all" if show_points else "outliers", labels={lab_metric: lab_metric_label})
            else:
                fig = px.violin(eeg_df, y=lab_metric, box=True, points="all" if show_points else "outliers", labels={lab_metric: lab_metric_label})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**What this panel shows (plain language):**")
    st.markdown(f"- The left panel visualizes `{lab_metric_label}` across the selected grouping (or overall). If you grouped by `condition`, you can directly compare Normal Sleep vs Sleep Deprived averages and distributions.")
    st.info("Reminder: Lab and survey datasets are *not* linked at the participant level. Comparisons are conceptual and exploratory.")

# Right: NHIS visualization
with col2:
    st.subheader("Survey (NHIS) — distribution by demographic")
    st.markdown(f"**Metric:** `{survey_metric}` — *{survey_metric_label}*")

    if nhis_group_by:
        if nhis_df[nhis_group_by].dtype != object:
            nhis_df[nhis_group_by] = nhis_df[nhis_group_by].astype(str)
        group_df2 = nhis_df.groupby(nhis_group_by)[survey_metric].agg([np.mean, np.std, np.median, "count"]).reset_index()
        group_df2 = group_df2.rename(columns={"mean":"mean", "std":"std", "median":"median", "count":"n"})
        st.write("Group summary:")
        st.dataframe(group_df2.sort_values("mean", ascending=False))

        if plot_kind == "bar":
            fig2 = px.bar(group_df2, x=nhis_group_by, y="mean", error_y="std", labels={nhis_group_by: labelize(nhis_group_by), "mean": survey_metric_label})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig2 = px.box(nhis_df, x=nhis_group_by, y=survey_metric, points="all" if show_points else "outliers", labels={nhis_group_by: labelize(nhis_group_by), survey_metric: survey_metric_label})
            if plot_kind == "violin":
                fig2 = px.violin(nhis_df, x=nhis_group_by, y=survey_metric, box=True, points="all" if show_points else "outliers", labels={nhis_group_by: labelize(nhis_group_by), survey_metric: survey_metric_label})
            st.plotly_chart(fig2, use_container_width=True)
    else:
        desc2 = nhis_df[survey_metric].describe()
        st.write("Global summary stats:")
        st.table(desc2.to_frame(name=survey_metric_label).T)
        if plot_kind == "bar":
            st.info("Bar requires grouping; using binned distribution instead.")
            binned2 = pd.cut(nhis_df[survey_metric].dropna(), bins=8)
            binned_counts2 = nhis_df.groupby(binned2)[survey_metric].count().rename("count").reset_index()
            fig2 = px.bar(binned_counts2, x=binned_counts2.index.astype(str), y="count", labels={"x":"Value bin", "count":"Count"})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            if plot_kind == "box":
                fig2 = px.box(nhis_df, y=survey_metric, points="all" if show_points else "outliers", labels={survey_metric: survey_metric_label})
            else:
                fig2 = px.violin(nhis_df, y=survey_metric, box=True, points="all" if show_points else "outliers", labels={survey_metric: survey_metric_label})
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**What this panel shows (plain language):**")
    st.markdown(f"- The right panel visualizes `{survey_metric_label}` across the selected demographic grouping. For example, if `age group` is selected, you can compare average sleep hours (or other sleep measures) across age groups.")

# -----------------------
# Conceptual comparison: distribution correlation + juxtaposition
# -----------------------
st.markdown("---")
st.header("Conceptual comparison & distribution correlation (exploratory)")

st.markdown("""
We cannot correlate person-level EEG and NHIS responses because datasets are not linked. 
Instead, this section creates *distribution-level* comparisons:
1. If you aggregated EEG by `condition` and NHIS by the same concept (or comparable group), you can compare group means.
2. We also compute a Spearman/Pearson correlation between the two selected metrics after coarse binning / resampling to produce a rough sense of co-movement across groups (conceptual only).
""")

# Strategy: create comparable group-level series if both grouped or otherwise use percentiles
def get_grouped_series(df, metric, group_by, target_bins=8, label_prefix=""):
    if group_by:
        if df[group_by].dtype != object:
            df[group_by] = df[group_by].astype(str)
        g = df.groupby(group_by)[metric].agg(["mean", "count"]).reset_index().rename(columns={"mean": "value"})
        g = g.sort_values("value")
        # return series of values
        s = g["value"].reset_index(drop=True)
        idx = g[group_by].astype(str).reset_index(drop=True)
        return s, idx
    else:
        # if no group: make binned means (e.g., quantiles)
        ser = df[metric].dropna()
        if ser.empty:
            return pd.Series(dtype=float), pd.Series(dtype=str)
        q = pd.qcut(ser, q=min(target_bins, len(ser.unique())), duplicates="drop")
        g = ser.groupby(q).mean().reset_index(name="value")
        idx = g[q.name].astype(str)
        return g["value"], idx

lab_series, lab_idx = get_grouped_series(eeg_df, lab_metric, eeg_group_by)
survey_series, survey_idx = get_grouped_series(nhis_df, survey_metric, nhis_group_by)

# align lengths by truncating to shortest
min_len = min(len(lab_series), len(survey_series))
lab_s_al = lab_series.iloc[:min_len]
surv_s_al = survey_series.iloc[:min_len]

r, p = safe_corr(lab_s_al, surv_s_al, method=correlation_method)
st.write(f"Correlation ({correlation_method}) between the selected *group-level* distributions: **r = {np.nan if np.isnan(r) else round(r,3)}**, p = {np.nan if np.isnan(p) else round(p,4)}.")
if np.isnan(r):
    st.info("Not enough data after grouping/aggregation to compute a reliable correlation. Try different groupings or metrics.")
else:
    st.markdown("- Interpretation (exploratory): small absolute r implies weak association across groups; this is not a person-level correlation and cannot support causal claims.")

# show a small juxtaposed line chart if lengths > 1
if min_len >= 2:
    juxtapose_df = pd.DataFrame({
        "EEG": lab_s_al.values,
        "NHIS": surv_s_al.values,
        "index": range(min_len)
    })
    fig_j = px.line(juxtapose_df, x="index", y=["EEG", "NHIS"], labels={"index":"Group index", "value":"Aggregated value"})
    st.plotly_chart(fig_j, use_container_width=True)

# -----------------------
# Download / data preview
# -----------------------
st.markdown("---")
st.header("Data preview & download")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Sample of EEG data used")
    st.dataframe(eeg_df[[lab_metric] + ([eeg_group_by] if eeg_group_by else [])].head(200))
    st.download_button("Download EEG CSV (filtered view)", eeg_df.to_csv(index=False), file_name="eeg_summary_export.csv")

with right_col:
    st.subheader("Sample of NHIS data used")
    st.dataframe(nhis_df[[survey_metric] + ([nhis_group_by] if nhis_group_by else [])].head(200))
    st.download_button("Download NHIS CSV (filtered view)", nhis_df.to_csv(index=False), file_name="nhis_sleep_export.csv")

# -----------------------
# Final notes and reproducibility
# -----------------------
st.markdown("---")
st.header("Notes, assumptions, and reproducibility")
st.markdown("""
- **Exploratory:** this tool is descriptive and educational. It does not link records across datasets and is **not** for diagnosis.
- **Assumptions:** data cleaning / definitions come from the repo. Check the `data/clean/` scripts for precise preprocessing (e.g., whether sleep hours are rounded, how missingness was handled).
- **Reproducibility:** all aggregation steps are visible in the app. For exact code integration, consider exporting the grouped DF to CSV and including the tidy pipeline in your repo's `notebooks/` folder.
""")






