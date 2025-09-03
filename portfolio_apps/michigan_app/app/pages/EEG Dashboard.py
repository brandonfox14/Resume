# -*- coding: utf-8 -*-
# EEG Dashboard – interpretable, accessible, explainable

import os
import json
from typing import Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="EEG Dashboard", layout="wide")
st.title("🧠 EEG Sleep Deprivation Explorer")

# =============================================================================
# Paths + loaders
# =============================================================================
DATA_CANDIDATES = [
    "data/clean/eeg_summary.csv",
    "data/eeg_summary.csv",
    "/mnt/data/eeg_summary.csv",
]
PARTICIPANTS_TSV_CANDIDATES = [
    "data/raw/eeg/participants.tsv",
    "data/participants.tsv",
    "/mnt/data/participants.tsv",
]
DATA_DICTIONARY_JSON = "/mnt/data/participants.json"  # optional: field tooltips

def _first_existing_path(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""

@st.cache_data
def load_summary() -> pd.DataFrame:
    p = _first_existing_path(DATA_CANDIDATES)
    if not p:
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    return df

@st.cache_data
def load_participants_tsv() -> pd.DataFrame:
    p = _first_existing_path(PARTICIPANTS_TSV_CANDIDATES)
    if not p:
        return pd.DataFrame()
    try:
        dfp = pd.read_csv(p, sep="\t")
    except Exception:
        try:
            dfp = pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    if "participant_id" in dfp.columns:
        dfp["participant_id"] = dfp["participant_id"].astype(str)
    return dfp

@st.cache_data
def load_data_dictionary() -> Dict[str, dict]:
    try:
        with open(DATA_DICTIONARY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

df = load_summary()
meta_df = load_participants_tsv()
data_dict = load_data_dictionary()

if df.empty:
    st.error("No EEG data found. I looked for: " + ", ".join(DATA_CANDIDATES))
    st.stop()

# =============================================================================
# Labels / palette / helpers
# =============================================================================
COND_LABELS = {"NS": "Normal Sleep (NS)", "SD": "Sleep Deprived (SD)"}
COND_FROM_LABEL = {v:k for k,v in COND_LABELS.items()}
TASK_LABELS = {"eyesopen": "Eyes Open", "eyesclosed": "Eyes Closed"}
PALETTE = {
    "NS": "#1f77b4",    # blue
    "SD": "#E69F00",    # orange
    "Theta": "#2ca02c", # green
    "Alpha": "#9467bd", # purple
    "Beta":  "#17becf", # teal
}

def tip(field: str, fallback: str = "") -> str:
    info = data_dict.get(field, {})
    desc = info.get("Description")
    return desc if desc else fallback

def kpi_card(title: str, value: str, help_text: str = ""):
    st.metric(title, value, help=help_text if help_text else None)

def melt_condition_wide(df_in: pd.DataFrame, base_names: List[str]) -> pd.DataFrame:
    """Turn columns like base_NS, base_SD into tidy rows with 'condition'."""
    cols = []
    for base in base_names:
        for suf in ["NS","SD"]:
            c = f"{base}_{suf}"
            if c in df_in.columns:
                cols.append((base, suf, c))
    if not cols:
        return pd.DataFrame()
    recs = []
    for _, row in df_in.iterrows():
        for base, suf, c in cols:
            recs.append({
                "participant_id": row.get("participant_id"),
                "condition": suf,
                "measure": base,
                "value": row[c]
            })
    return pd.DataFrame.from_records(recs)

def available(series: pd.Series) -> int:
    return int(series.notna().sum())

# =============================================================================
# Sidebar filters
# =============================================================================
st.sidebar.header("Filters")
# Condition
conds_present = [c for c in ["NS","SD"] if "condition" in df.columns and c in df["condition"].unique()]
if not conds_present:
    conds_present = ["NS","SD"]
user_conds = st.sidebar.multiselect(
    "Condition", [COND_LABELS[c] for c in conds_present],
    default=[COND_LABELS[c] for c in conds_present],
    help="Select one or both sleep conditions."
)
conds_selected = [COND_FROM_LABEL[c] for c in user_conds] if user_conds else conds_present

# Task
tasks_present = [t for t in ["eyesopen","eyesclosed"] if "task" in df.columns and t in df["task"].unique()]
if not tasks_present:
    tasks_present = ["eyesopen","eyesclosed"]
user_tasks = st.sidebar.multiselect(
    "Task", [TASK_LABELS[t] for t in tasks_present],
    default=[TASK_LABELS[t] for t in tasks_present],
    help="Select eyes-open and/or eyes-closed trials."
)
tasks_selected = [k for k,v in TASK_LABELS.items() if v in user_tasks] if user_tasks else tasks_present

# Demographics
age_min, age_max = (int(df["Age"].min()), int(df["Age"].max())) if "Age" in df.columns and df["Age"].notna().any() else (18, 80)
age_range = st.sidebar.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max), help=tip("Age","Participant age in years."))
gender_vals = sorted([g for g in df.get("Gender", pd.Series(dtype=str)).dropna().unique().tolist() if g])
gender_sel = st.sidebar.multiselect("Sex", gender_vals, default=gender_vals, help=tip("Gender","Biological sex as recorded."))

# Apply filters to rows (session×task)
mask = pd.Series(True, index=df.index)
if "condition" in df.columns:
    mask &= df["condition"].isin(conds_selected)
if "task" in df.columns:
    mask &= df["task"].isin(tasks_selected)
if "Age" in df.columns:
    mask &= df["Age"].between(age_range[0], age_range[1])
if "Gender" in df.columns and gender_vals:
    mask &= df["Gender"].isin(gender_sel) if gender_sel else True
df_filt = df.loc[mask].copy()

# A de-duplicated participant slice for cohort-level KPIs (avoid double-counting)
people = (df_filt.sort_values("participant_id")
                 .drop_duplicates(subset="participant_id"))

# =============================================================================
# Data quality header
# =============================================================================
st.subheader("Data quality snapshot")
cards = st.columns(5)
with cards[0]:
    kpi_card("Participants", f"{people['participant_id'].nunique()}")
with cards[1]:
    kpi_card("Records (rows)", f"{len(df_filt)}", "One row = one session×task")
with cards[2]:
    kpi_card("Theta avail", f"{available(df_filt.get('theta_mean', pd.Series(dtype=float)))}")
with cards[3]:
    kpi_card("Alpha avail", f"{available(df_filt.get('alpha_mean', pd.Series(dtype=float)))}")
with cards[4]:
    kpi_card("Beta avail",  f"{available(df_filt.get('beta_mean',  pd.Series(dtype=float)))}")

needs = ["theta_mean","alpha_mean","beta_mean"]
if any(c not in df_filt.columns for c in needs):
    st.info("Some EEG band columns are missing from this dataset.")
else:
    missing_rows = df_filt[df_filt[needs].isna().any(axis=1)]
    if len(missing_rows) == 0:
        st.success("All selected rows include theta, alpha, and beta.")
    else:
        st.warning(f"{len(missing_rows)} selected row(s) are missing one or more bands.")
        with st.expander("Show missing rows"):
            st.dataframe(
                missing_rows[["participant_id","session","task","condition","theta_mean","alpha_mean","beta_mean"]],
                use_container_width=True
            )

st.divider()

# =============================================================================
# Why these metrics? (short rationale)
# =============================================================================
st.markdown("**Before we dive into the charts, here’s the quick ‘why’ behind the metrics you’ll see.**")
with st.expander("Why these metrics?", expanded=False):
    st.markdown(
        "- **PANAS** shows how mood states (positive/negative) can shift with sleep loss.  \n"
        "- **PVT** captures vigilant attention; sleep deprivation often slows reaction time and increases lapses.  \n"
        "- **EEG bands** connect behavior to physiology: alpha tends to drop with eyes open; theta can rise as alertness falls."
    )

# =============================================================================
# Mood (PANAS)
# =============================================================================
st.markdown("### Mood (PANAS)")
with st.expander("What are Positive and Negative Affect?", expanded=False):
    st.markdown(
        "**Positive Affect (PA)**: feeling **enthusiastic/energetic** (e.g., *interested, excited, inspired*). "
        "Higher PA → more positive mood.  \n"
        "**Negative Affect (NA)**: feeling **distressed/irritable** (e.g., *upset, guilty, scared*). "
        "Higher NA → more negative mood.  \n"
        "We treat them as **state** ratings around testing—useful for seeing how sleep changes mood."
    )
with st.expander("How to read this chart", expanded=False):
    st.markdown(
        "- **Box** = middle 50% of scores; **line** = median.  \n"
        "- **Whiskers** show typical low/high values.  \n"
        "- **Dots** = one dot per participant (jittered sideways so they don’t overlap).  \n"
        "- Colors: **blue = NS**, **orange = SD**."
    )
with st.expander("Why this chart?", expanded=False):
    st.markdown(
        "A **box plot** shows group differences and spread **without assuming normality**. "
        "Overlaying dots keeps individuals visible (great for small samples)."
    )

panas_tidy = melt_condition_wide(df_filt, ["PANAS_P","PANAS_N"])
if not panas_tidy.empty:
    col_choice = st.radio(
        "Which mood scores to view?",
        ["Positive Affect (PA)", "Negative Affect (NA)"],
        horizontal=True
    )
    which = "PANAS_P" if col_choice.startswith("Positive") else "PANAS_N"
    pdat = panas_tidy[panas_tidy["measure"] == which].dropna(subset=["value"])
    if pdat.empty:
        st.info("No PANAS values available under current filters.")
    else:
        fig = px.box(
            pdat, x="condition", y="value",
            color="condition",
            color_discrete_map={"NS": PALETTE["NS"], "SD": PALETTE["SD"]},
            points="all", hover_data=["participant_id"],
            labels={"value":"Score","condition":"Condition"}
        )
        fig.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each dot is one participant’s score under that condition.")

        # Quick summary (means)
        g = pdat.groupby("condition")["value"].mean()
        if "NS" in g and "SD" in g:
            diff = g["SD"] - g["NS"]
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            st.caption(f"**Summary** - Mean {col_choice.split()[0]}: NS = {g['NS']:.1f}, SD = {g['SD']:.1f}. SD–NS: {diff:+.1f} ({arrow}).")
else:
    st.caption("PANAS fields not found (expected columns like PANAS_P_NS, PANAS_N_SD).")

st.divider()

# =============================================================================
# Psychomotor Vigilance Test (PVT)
# =============================================================================
st.markdown("### Vigilant attention: Psychomotor Vigilance Test (PVT)")
with st.expander("About the Psychomotor Vigilance Test", expanded=False):
    st.markdown(
        "The **Psychomotor Vigilance Test (PVT)** is a widely used reaction-time task in sleep research. "
        "Participants respond as quickly as possible to repeating visual cues over several minutes.  \n"
        "- **Lapses**: very slow responses (often > 500 ms). Fewer lapses indicate better sustained attention.  \n"
        "- **Median RT**: the typical reaction time in milliseconds; lower values mean faster responses.  \n"
        "- **RT variability**: the spread of reaction times (standard deviation); a smaller spread means more consistent performance."
    )
with st.expander("How to read this chart", expanded=False):
    st.markdown(
        "- The **violin shape** shows the full distribution of scores, including any very slow lapses.  \n"
        "- The **white box** marks the middle 50% of values (IQR) and the horizontal line shows the median.  \n"
        "- **Dots** are individual participants (slightly jittered left/right to avoid overlap).  \n"
        "- **Blue** = Normal Sleep (NS), **Orange** = Sleep Deprived (SD)."
    )
with st.expander("Why use a violin plot here?", expanded=False):
    st.markdown(
        "Sleep loss often produces a wide spread of reaction times with heavy tails due to lapses. "
        "A **violin plot** shows both the distribution shape and summary stats, which a box alone can hide."
    )

pvt_tidy = melt_condition_wide(df_filt, ["PVT_item1","PVT_item2","PVT_item3"])
if not pvt_tidy.empty:
    metric_map = {
        "Lapses (count)": "PVT_item1",
        "Median RT (ms)": "PVT_item2",
        "RT variability (SD, ms)": "PVT_item3",
    }
    which_label = st.radio("Metric", list(metric_map.keys()), horizontal=True)
    which_key = metric_map[which_label]
    pdata = pvt_tidy[pvt_tidy["measure"] == which_key].dropna(subset=["value"])
    if pdata.empty:
        st.info("No PVT values available under current filters.")
    else:
        fig = px.violin(
            pdata, x="condition", y="value", color="condition",
            color_discrete_map={"NS": PALETTE["NS"], "SD": PALETTE["SD"]},
            box=True, points="all", hover_data=["participant_id"],
            labels={"value": which_label, "condition":"Condition"}
        )
        fig.update_layout(height=440, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each dot is one participant’s value under that condition.")

        g = pdata.groupby("condition")["value"].mean()
        if "NS" in g and "SD" in g:
            diff = g["SD"] - g["NS"]
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            st.caption(f"**Summary** - Mean {which_label}: NS = {g['NS']:.0f}, SD = {g['SD']:.0f}. SD–NS: {diff:+.0f} ({arrow}).")
else:
    st.caption("PVT fields not found (expected columns like PVT_item2_NS / _SD).")

st.divider()

# =============================================================================
# EEG band power
# =============================================================================
st.markdown("### EEG band power")
with st.expander("Band definitions", expanded=False):
    st.markdown(
        "- **Theta**: 4–7 Hz; often increases with reduced alertness.  \n"
        "- **Alpha**: 8–12 Hz; typically higher with eyes closed; decreases with eyes open.  \n"
        "- **Beta**: 13–30 Hz; associated with alert/cognitive engagement."
    )
with st.expander("How to read this chart", expanded=False):
    st.markdown(
        "Grouped **box plots** compare bands by condition (or by task if you toggle it). "
        "Higher values = more power in that frequency band. **Dots** show individual participants."
    )
with st.expander("Why this chart?", expanded=False):
    st.markdown(
        "Box plots compactly compare **central tendency and variability** across bands/conditions, "
        "and are robust to non‑normal data."
    )

band_cols = [c for c in ["theta_mean","alpha_mean","beta_mean"] if c in df_filt.columns]
if band_cols:
    split_by_task = st.checkbox("Split by task (eyes open vs. eyes closed)", value=False)
    if split_by_task and "task" in df_filt.columns:
        fig = px.box(
            df_filt.melt(
                id_vars=["participant_id","condition","task"],
                value_vars=band_cols, var_name="Band", value_name="Power (dB)"
            ).dropna(subset=["Power (dB)"]),
            x="Band", y="Power (dB)", color="task", facet_col="condition",
            color_discrete_map={"eyesopen":"#636EFA", "eyesclosed":"#EF553B"},
            category_orders={"Band": ["theta_mean","alpha_mean","beta_mean"]},
            labels={"task":"Task","Band":""}
        )
    else:
        fig = px.box(
            df_filt.melt(
                id_vars=["participant_id","condition"],
                value_vars=band_cols, var_name="Band", value_name="Power (dB)"
            ).dropna(subset=["Power (dB)"]),
            x="Band", y="Power (dB)", color="condition",
            color_discrete_map={"NS": PALETTE["NS"], "SD": PALETTE["SD"]},
            labels={"Band":""}
        )
    fig.for_each_xaxis(lambda ax: ax.update(categoryorder="array",
                                            categoryarray=["theta_mean","alpha_mean","beta_mean"],
                                            ticktext=["Theta (4–7 Hz)","Alpha (8–12 Hz)","Beta (13–30 Hz)"],
                                            tickvals=["theta_mean","alpha_mean","beta_mean"]))
    fig.update_layout(height=460, boxmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Each dot is one participant’s band power value.")
else:
    st.info("No EEG band columns found.")

st.divider()

# =============================================================================
# Cohort characteristics & Participants table
# =============================================================================
st.subheader("Cohort characteristics")
ccols = st.columns(4)
with ccols[0]:
    mean_age = people["Age"].mean() if "Age" in people.columns else np.nan
    kpi_card("Mean age", f"{mean_age:.1f}" if pd.notna(mean_age) else "-")
with ccols[1]:
    f_count = people.loc[people["Gender"].eq("F")] if "Gender" in people.columns else pd.DataFrame()
    kpi_card("Female participants", f"{f_count['participant_id'].nunique() if not f_count.empty else 0}")
with ccols[2]:
    m_count = people.loc[people["Gender"].eq("M")] if "Gender" in people.columns else pd.DataFrame()
    kpi_card("Male participants", f"{m_count['participant_id'].nunique() if not m_count.empty else 0}")
with ccols[3]:
    sess_orders = ", ".join(sorted(people.get("SessionOrder", pd.Series(dtype=str)).dropna().unique().tolist()))
    kpi_card("Session orders", sess_orders if sess_orders else "-")

st.markdown("#### Participants")
view_mode = st.radio("View mode", ["Compact (one row per participant)", "Long (session × task)"], horizontal=True)

def build_availability_grid(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return one row per participant + four availability slots (NS/SD × eyes open/closed)."""
    needed = {"participant_id","session","task","condition"}
    cols_base = [c for c in ["participant_id","Gender","Age","SessionOrder"] if c in df_in.columns]
    if not needed.issubset(df_in.columns):
        return df_in[cols_base].drop_duplicates().sort_values("participant_id")

    df_small = df_in[cols_base + ["session","task","condition"]].copy()
    df_small["slot"] = df_small["condition"].map({"NS":"NS","SD":"SD"}) + " • " + df_small["task"].map({"eyesopen":"eyes open","eyesclosed":"eyes closed"})
    slots = ["NS • eyes closed","NS • eyes open","SD • eyes closed","SD • eyes open"]

    pivot = (df_small.assign(available="✓")
             .drop_duplicates(["participant_id","slot"])
             .pivot_table(index=cols_base, columns="slot", values="available", aggfunc="first")
             .reindex(columns=slots))
    pivot = pivot.reset_index()
    pivot.columns.name = None

    for c in slots:
        if c not in pivot.columns:
            pivot[c] = ""
    return pivot[cols_base + slots].sort_values("participant_id")

if view_mode.startswith("Compact"):
    compact_df = build_availability_grid(df_filt)
    st.dataframe(compact_df, use_container_width=True)
    st.download_button(
        "Download (compact CSV)",
        data=compact_df.to_csv(index=False).encode("utf-8"),
        file_name="participants_compact.csv",
        mime="text/csv"
    )
else:
    show_cols = [c for c in ["participant_id","Gender","Age","SessionOrder","session","task","condition"] if c in df_filt.columns]
    st.dataframe(df_filt[show_cols].sort_values(["participant_id","session","task"]), use_container_width=True)
    st.download_button(
        "Download (long CSV)",
        data=df_filt.to_csv(index=False).encode("utf-8"),
        file_name="eeg_summary_filtered.csv",
        mime="text/csv"
    )

st.divider()
