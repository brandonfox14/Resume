# -*- coding: utf-8 -*-
# EEG Channel Explorer - CSV-first coordinates + SVG fallback
# Uses: data/clean/eeg_channel_coordinates.csv

import os
import re
import base64
import xml.etree.ElementTree as ET

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download, list_repo_files

# =============================================================================
# App setup
# =============================================================================
HF_REPO = "aparker03/eeg-csv"
DATA_DIR = "data/eeg_csv"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="EEG Channel Explorer", layout="wide")
st.title("🧠 EEG Channel Explorer")

# =============================================================================
# Utility
# =============================================================================
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# =============================================================================
# Pretty labels / titles
# =============================================================================
def pretty_condition_suffix(suf: str) -> str:
    return {"NS": "Normal Sleep", "SD": "Sleep Deprived"}.get(suf, suf)

def pretty_title(subj_id: str, cond_label: str, task_label: str) -> str:
    subj_num = subj_id.replace("sub-", "")
    cond = "Normal Sleep" if "Normal" in cond_label else "Sleep Deprived"
    return f"Subject {subj_num} • {cond} • {task_label}"

def pretty_metric_name(col: str) -> str:
    if col.startswith("PANAS_P_"):
        return f"Positive Affect (PANAS) - {pretty_condition_suffix(col.split('_')[-1])}"
    if col.startswith("PANAS_N_"):
        return f"Negative Affect (PANAS) - {pretty_condition_suffix(col.split('_')[-1])}"
    if col.startswith("PVT_item1_"):
        return f"PVT Lapses (count) - {pretty_condition_suffix(col.split('_')[-1])}"
    if col.startswith("PVT_item2_"):
        return f"PVT Median Reaction Time (ms) - {pretty_condition_suffix(col.split('_')[-1])}"
    if col.startswith("PVT_item3_"):
        return f"PVT RT Variability (SD, ms) - {pretty_condition_suffix(col.split('_')[-1])}"
    return col

def metric_help(col: str) -> str:
    if col.startswith("PANAS_P_"):
        return "PANAS Positive Affect: momentary positive mood. Higher is more positive."
    if col.startswith("PANAS_N_"):
        return "PANAS Negative Affect: momentary negative mood. Higher is more negative."
    if col.startswith("PVT_item1_"):
        return "PVT lapses: very slow or missed responses. Lower is better."
    if col.startswith("PVT_item2_"):
        return "PVT median reaction time (ms). Lower is better."
    if col.startswith("PVT_item3_"):
        return "PVT RT variability (SD, ms). Lower is more consistent."
    return ""

# =============================================================================
# Background SVG (visual only)
# =============================================================================
SVG_CANDIDATES = [
    "assets/brain_map.svg",
    "/mnt/data/brain_map.svg",
    "/mnt/data/International_10-20_system_for_EEG-MCN.svg",
]

def _load_svg_bytes():
    for p in SVG_CANDIDATES:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read(), p
    return None, None

SVG_BYTES, SVG_PATH = _load_svg_bytes()

def svg_to_data_uri(svg_bytes: bytes) -> str:
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"

CHANNEL_RX = re.compile(
    r"^(?:"
    r"Fpz|Fp1|Fp2|"
    r"AFz|AF[3-8]|"
    r"Fz|F[1-8]|"
    r"FCz|FC[1-6]|"
    r"Cz|C[1-6]|"
    r"CPz|CP[1-6]|"
    r"Pz|P[1-8]|"
    r"POz|PO[3-8]|"
    r"Oz|O[1-2]|"
    r"T[78]|T9|T10|"
    r"FT[78]|"
    r"TP[78]|TP9|TP10"
    r")$",
    re.I
)

def _norm_label(s: str) -> str:
    return str(s).strip().upper().replace(" ", "")

def parse_svg_channel_positions(svg_bytes: bytes):
    if not svg_bytes:
        return {}
    try:
        root = ET.fromstring(svg_bytes)
    except Exception:
        return {}

    vb = root.attrib.get("viewBox")
    if vb:
        minx, miny, width, height = [float(v) for v in vb.strip().split()]
    else:
        width = float(root.attrib.get("width", "1000").replace("px", "") or 1000)
        height = float(root.attrib.get("height", "1000").replace("px", "") or 1000)
        minx, miny = 0.0, 0.0

    ns = {"svg": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    def _fp(tag):
        return f"{{{ns['svg']}}}{tag}" if ns else tag

    positions = {}

    # 1) text/tspan
    for t in root.iter():
        if t.tag not in (_fp("text"), _fp("tspan")):
            continue
        label = (t.text or "").strip()
        if not label or len(label) > 4 or not CHANNEL_RX.match(label):
            continue
        x = t.attrib.get("x"); y = t.attrib.get("y")
        if x is None or y is None:
            continue
        try:
            x = float(x); y = float(y)
        except Exception:
            continue
        X = (x - minx) / width
        Y = 1.0 - ((y - miny) / height)
        positions[_norm_label(label)] = (X, Y)

    # 2) id/class/data-*
    def harvest_from_attrs(elem):
        for attr in ("id", "class", "data-label", "data-name"):
            val = elem.attrib.get(attr)
            if not val:
                continue
            for token in re.split(r"[\s,;:]+", val):
                tok = token.strip()
                if CHANNEL_RX.match(tok):
                    cx = elem.attrib.get("cx"); cy = elem.attrib.get("cy")
                    x = elem.attrib.get("x");  y = elem.attrib.get("y")
                    use_x = use_y = None
                    if cx and cy:
                        try:
                            use_x = float(cx); use_y = float(cy)
                        except Exception:
                            pass
                    elif x and y:
                        try:
                            use_x = float(x); use_y = float(y)
                        except Exception:
                            pass
                    if use_x is not None and use_y is not None:
                        X = (use_x - minx) / width
                        Y = 1.0 - ((use_y - miny) / height)
                        positions[_norm_label(tok)] = (X, Y)

    for e in root.iter():
        harvest_from_attrs(e)

    if not positions:
        return {}

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    dx = max(x1 - x0, 1e-9)
    dy = max(y1 - y0, 1e-9)

    cropped = {}
    for lbl, (X, Y) in positions.items():
        cx = (X - x0) / dx
        cy = (Y - y0) / dy
        cropped[lbl] = (cx, cy)

    return cropped

SVG_POS_RAW = parse_svg_channel_positions(SVG_BYTES)
SVG_POS = {k: (x, y) for k, (x, y) in SVG_POS_RAW.items()}

# =============================================================================
# Channel coordinate CSV (PRIMARY)
# =============================================================================
COORDS_PATH = "data/clean/eeg_channel_coordinates.csv"

@st.cache_data
def _load_coords_csv(path: str):
    if not os.path.exists(path):
        return {}, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, None
    label_col = next((c for c in df.columns if str(c).lower() in ["label","channel","chan","name"]), None)
    x_col     = next((c for c in df.columns if str(c).lower() in ["x","cx","xcoord","x_pos"]), None)
    y_col     = next((c for c in df.columns if str(c).lower() in ["y","cy","ycoord","y_pos"]), None)
    if not (label_col and x_col and y_col):
        return {}, None

    xmin, xmax = df[x_col].min(), df[x_col].max()
    ymin, ymax = df[y_col].min(), df[y_col].max()
    dx = max(xmax - xmin, 1e-9); dy = max(ymax - ymin, 1e-9)

    pos = {}
    for _, r in df.iterrows():
        try:
            xn = (float(r[x_col]) - xmin) / dx
            yn = 1.0 - ((float(r[y_col]) - ymin) / dy)  # flip y for plotly
        except Exception:
            continue
        pos[_norm_label(r[label_col])] = (xn, yn)
    return pos, (label_col, x_col, y_col)

COORD_POS, COORD_COLUMNS = _load_coords_csv(COORDS_PATH)
USE_COORDS = bool(COORD_POS)

def get_xy(label: str):
    if not label:
        return None
    key = _norm_label(label)
    if USE_COORDS and key in COORD_POS:
        return COORD_POS[key]
    return SVG_POS.get(key)

# =============================================================================
# Region membership and colors
# =============================================================================
REGION_MAP = {
    "Frontal":   ["Fp1","Fp2","Fpz","AFz","AF3","AF4","AF7","AF8","F1","F2","F3","F4","F7","F8","Fz"],
    "Central":   ["FC1","FC2","FCz","C1","C2","C3","C4","Cz"],
    "Parietal":  ["CP1","CP2","CPz","P1","P2","P3","P4","P7","P8","Pz","POz"],
    "Occipital": ["PO3","PO4","PO7","PO8","O1","Oz","O2"],
    "Temporal":  ["T7","T8","T9","T10","FT7","FT8","TP7","TP8","TP9","TP10"],
}
# High-contrast, color-blind–friendly choices
REGION_COLOR = {
    "Frontal":   "#1f77b4",  # blue
    "Central":   "#E69F00",  # golden orange (distinct)
    "Parietal":  "#2ca02c",  # green
    "Occipital": "#9467bd",  # purple
    "Temporal":  "#17becf",  # teal (clearly different from Central)
}

# =============================================================================
# Region hull (for soft highlight polygons)
# =============================================================================
def _hull(points):
    pts = sorted(points)
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def region_polygon(region):
    chs = REGION_MAP.get(region, [])
    pts = [get_xy(c) for c in chs if get_xy(c) is not None]
    if len(pts) < 3:
        return None
    hull = _hull(pts)
    xs, ys = zip(*hull)
    return xs, ys

# =============================================================================
# Channel and region explainers
# =============================================================================
REGION_TEXT = {
    "Frontal": (
        "Front of the head. Often related to attention, working memory, and eye movement artifacts.",
        "International 10–20 frontal leads. Common focus in executive function studies."
    ),
    "Central": (
        "Top center. Near sensorimotor cortex. Often shows rhythms linked to movement and touch.",
        "Central midline and peri‑central electrodes. Used for mu/beta rhythms and motor tasks."
    ),
    "Parietal": (
        "Upper back of the head. Linked to attention and integration of sensory info.",
        "Parietal electrodes. Often used in attention and visuospatial tasks."
    ),
    "Occipital": (
        "Back of the head. Strong visual signals. Alpha increases with eyes closed.",
        "Occipital leads. Classic site for alpha rhythm and visual processing."
    ),
    "Temporal": (
        "Sides of the head near the ears. Involved in audition and language.",
        "Temporal and temporo‑parietal leads. Used in language and auditory studies."
    ),
}

def channel_explainer(label: str):
    L = label.upper()
    if L.startswith("FP"):
        return ("Frontal pole. Forehead area; can capture eye movement artifacts.",
                "Frontal‑polar lead in the 10–20 system.")
    if L.startswith("AF") or L.startswith("F"):
        return ("Frontal region. Linked to attention and executive control.",
                "Anterior‑frontal or frontal lead.")
    if L.startswith("FC") or L.startswith("C"):
        return ("Central region. Near sensorimotor cortex.",
                "Frontocentral or central lead over/near central sulcus.")
    if L.startswith("CP") or L.startswith("P"):
        return ("Parietal region. Sensory integration and attention.",
                "Centro‑parietal or parietal lead.")
    if L.startswith("PO") or L.startswith("O"):
        return ("Occipital region. Visual processing; strong alpha when eyes closed.",
                "Parieto‑occipital or occipital lead.")
    if L.startswith("T") or L.startswith("TP") or L.startswith("FT"):
        return ("Temporal region. Auditory and language processing.",
                "Temporal or temporo‑parietal lead.")
    return ("Standard 10–20 scalp position.", "Conventional montage label.")

# =============================================================================
# Brain map renderer
# =============================================================================
def build_brain_map(selected_channels, regions_to_draw, svg_bytes):
    fig = go.Figure()
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")

    if svg_bytes:
        fig.add_layout_image(
            dict(source=svg_to_data_uri(svg_bytes), xref="x", yref="y",
                 x=0, y=1, sizex=1, sizey=1, sizing="stretch",
                 layer="below", opacity=1.0)
        )

    # Region bands (no dots unless channels selected)
    for region in regions_to_draw:
        poly = region_polygon(region)
        if not poly:
            continue
        px, py = poly
        fig.add_trace(go.Scatter(
            x=list(px)+[px[0]], y=list(py)+[py[0]],
            mode="lines", fill="toself",
            line=dict(width=0),
            fillcolor=hex_to_rgba(REGION_COLOR[region], 0.22),
            hoverinfo="skip",
            showlegend=False
        ))

    # Dots only for selected channels
    if selected_channels:
        def color_for_channel(ch):
            for rname, chs in REGION_MAP.items():
                if ch in chs:
                    return REGION_COLOR[rname]
            return "#111111"
        xs, ys, names, outlines = [], [], [], []
        for ch in selected_channels:
            xy = get_xy(ch)
            if not xy:
                continue
            xs.append(xy[0]); ys.append(xy[1]); names.append(ch); outlines.append(color_for_channel(ch))
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=20, color="rgba(0,0,0,0.12)", line=dict(width=0)),
                hoverinfo="skip", showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=15, color="rgba(0,0,0,0)",
                            line=dict(color=outlines, width=3)),
                text=names, hoverinfo="text", showlegend=False
            ))

    fig.add_annotation(x=0.5, y=1.04, xref="x", yref="y",
                       text="FRONT (nasion)", showarrow=False)
    fig.add_annotation(x=0.5, y=-0.04, xref="x", yref="y",
                       text="BACK (inion)", showarrow=False)

    fig.update_xaxes(visible=False, range=[0,1])
    fig.update_yaxes(visible=False, range=[0,1], scaleanchor="x", scaleratio=1)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=10))
    return fig

# =============================================================================
# Session state
# =============================================================================
if "selected_subj" not in st.session_state:
    st.session_state.selected_subj = None
if "channel_sel" not in st.session_state:
    st.session_state.channel_sel = []
if "selected_regions" not in st.session_state:
    st.session_state.selected_regions = set()
if "region_mode" not in st.session_state:
    st.session_state.region_mode = "Single"  # Single | Multi | All

# =============================================================================
# Participants metadata
# =============================================================================
def _first_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

_PARTICIPANTS_PATH = _first_existing_path([
    "data/raw/eeg/participants.tsv",
    "data/participants.tsv"
])

@st.cache_data
def load_participants(path):
    if not path:
        return pd.DataFrame()
    try:
        dfp = pd.read_csv(path, sep="\t")
    except Exception:
        dfp = pd.read_csv(path)
    if "participant_id" in dfp.columns:
        dfp["participant_id_norm"] = (
            dfp["participant_id"].astype(str).str.replace("sub-", "", regex=False)
        )
    for col in [c for c in dfp.columns if "EEG_SamplingTime" in c]:
        dfp[col + "_hour"] = pd.to_datetime(dfp[col], errors="coerce", format="%H:%M").dt.hour
    return dfp

meta = load_participants(_PARTICIPANTS_PATH)

# =============================================================================
# Index remote EEG files
# =============================================================================
try:
    all_files = list_repo_files(repo_id=HF_REPO, repo_type="dataset")
except Exception:
    all_files = []

csv_files = [f for f in all_files if f.endswith(".csv")]
if not csv_files:
    st.error("No EEG CSV files found in the remote dataset. Check the Hugging Face path or your connection.")
    st.stop()

all_subject_ids = sorted({f.split("_")[0].replace("sub-", "") for f in csv_files})

def available_for_subject(subj_id: str):
    prefix = f"sub-{subj_id}_"
    avail = {}
    for f in csv_files:
        if f.startswith(prefix):
            parts = f.replace(prefix, "").split(".csv")[0].split("_")
            if len(parts) == 2:
                session, task = parts
                avail[(session, task)] = f
    return avail

# =============================================================================
# Sidebar: Participant Finder
# =============================================================================
st.sidebar.header("Find a participant")
st.sidebar.caption("Use filters to narrow the subject list. The viewer shows one subject at a time.")
show_group = st.sidebar.checkbox("Show group snapshot while filtering", value=False)

eligible_subjects = all_subject_ids.copy()

if not meta.empty and "participant_id_norm" in meta.columns:
    # Sex
    if "Gender" in meta.columns:
        all_sexes = sorted([s for s in meta["Gender"].dropna().unique().tolist() if s])
        sex_sel = st.sidebar.multiselect("Sex", all_sexes, default=all_sexes)
    else:
        sex_sel = None

    # Age
    if "Age" in meta.columns and meta["Age"].notna().any():
        amin = int(meta["Age"].min()); amax = int(meta["Age"].max())
        age_sel = st.sidebar.slider("Age (years)", amin, amax, (amin, amax))
    else:
        age_sel = None

    # Session order
    if "SessionOrder" in meta.columns:
        orders = [o for o in meta["SessionOrder"].dropna().unique().tolist() if o]
        order_sel = st.sidebar.multiselect("Session order", orders, default=orders)
    else:
        order_sel = None

# Condition / Task pickers
selected_cond = st.radio("🛌 Select Condition:", ["Normal Sleep (NS)", "Sleep Deprived (SD)"])
selected_task = st.radio("👁️ Select Task:", ["Eyes Open", "Eyes Closed"])

# Filter eligible subjects
if not meta.empty and "participant_id_norm" in meta.columns:
    m = pd.Series(True, index=meta.index)
    if "Gender" in meta.columns and sex_sel is not None:
        m &= meta["Gender"].isin(sex_sel)
    if "Age" in meta.columns and age_sel is not None:
        m &= meta["Age"].between(age_sel[0], age_sel[1])
    if "SessionOrder" in meta.columns and order_sel is not None:
        m &= meta["SessionOrder"].isin(order_sel)
    eligible_ids = meta.loc[m, "participant_id_norm"].astype(str).unique().tolist()
    eligible_subjects = sorted([s for s in all_subject_ids if s in set(eligible_ids)])

if not eligible_subjects:
    st.info("No participants match your filters. Try widening the age range or clearing a filter above.")
    st.stop()

# Subject picker (sticky)
prev = st.session_state.selected_subj
default_index = eligible_subjects.index(prev) if prev in eligible_subjects else 0
selected_subj = st.selectbox("👤 Select Subject:", eligible_subjects, index=default_index)
st.session_state.selected_subj = selected_subj
if prev and prev != selected_subj and prev not in eligible_subjects:
    st.info(f"Your previous selection {prev} is not in the filtered list now. Showing {selected_subj} instead.")

# Optional cohort snapshot
if show_group and not meta.empty and "participant_id_norm" in meta.columns:
    with st.expander("Group snapshot (for current filters)"):
        current_pool = meta[meta["participant_id_norm"].isin(eligible_subjects)].copy()
        n = len(current_pool)
        if n == 0:
            st.info("No matching participants.")
        else:
            cols = st.columns(3)
            cols[0].metric("Participants", n)
            if "Age" in current_pool.columns and current_pool["Age"].notna().any():
                cols[1].metric("Mean age (years)", f"{round(current_pool['Age'].mean(), 1)}")
            if "Gender" in current_pool.columns:
                g_counts = current_pool["Gender"].value_counts().to_dict()
                cols[2].metric("Sex counts", ", ".join(f"{k}:{v}" for k, v in g_counts.items()))
            suf = "NS" if selected_cond == "Normal Sleep (NS)" else "SD"
            metric_cols = [f"PANAS_P_{suf}", f"PANAS_N_{suf}",
                           f"PVT_item1_{suf}", f"PVT_item2_{suf}", f"PVT_item3_{suf}"]
            available = [c for c in metric_cols if c in current_pool.columns and current_pool[c].notna().any()]
            for i in range(0, len(available), 3):
                row = st.columns(3)
                for j, col in enumerate(available[i:i+3]):
                    row[j].metric(pretty_metric_name(col), round(current_pool[col].mean(), 2), help=metric_help(col))
        st.caption("Descriptive only. Patterns reflect this dataset; not clinical advice.")

# =============================================================================
# Region explainer and selection controls (no multiselect yet)
# =============================================================================
st.markdown("### Channels to plot")

mode = st.radio("Region selection mode:", ["Single", "Multi", "All"], horizontal=True,
                index=["Single","Multi","All"].index(st.session_state.region_mode))
st.session_state.region_mode = mode

def _toggle_region(region):
    if st.session_state.region_mode == "All":
        st.session_state.selected_regions = set(REGION_MAP.keys())
    elif st.session_state.region_mode == "Single":
        if region in st.session_state.selected_regions and len(st.session_state.selected_regions) == 1:
            st.session_state.selected_regions = set()
        else:
            st.session_state.selected_regions = {region}
    else:  # Multi
        if region in st.session_state.selected_regions:
            st.session_state.selected_regions.remove(region)
        else:
            st.session_state.selected_regions.add(region)

bcols = st.columns(6)
for i, region in enumerate(["Frontal","Central","Parietal","Occipital","Temporal"]):
    picked = region in st.session_state.selected_regions
    label = f"{'✅ ' if picked else ''}{region}"
    bcols[i].button(label, key=f"btn_{region}", on_click=_toggle_region, args=(region,))
# Reset clears regions and channels
bcols[5].button("Reset", key="btn_reset", on_click=lambda: (
    st.session_state.selected_regions.clear(),
    st.session_state.update({"channel_sel": []})
))

with st.expander("🗺 About EEG Regions", expanded=False):
    for r in ["Frontal","Central","Parietal","Occipital","Temporal"]:
        plain, tech = REGION_TEXT[r]
        st.markdown(f"**{r}**  \n{plain}  \n*{tech}*")

# =============================================================================
# Pick file for this subject / condition / task (load recording BEFORE channel picker)
# =============================================================================
cond_map = {"Normal Sleep (NS)": "ses-1", "Sleep Deprived (SD)": "ses-2"}
task_map = {"Eyes Open": "eyesopen", "Eyes Closed": "eyesclosed"}

subj_str = f"sub-{selected_subj}"
session_str = cond_map[selected_cond]
task_str = task_map[selected_task]
filename = f"{subj_str}_{session_str}_{task_str}.csv"
file_path = os.path.join(DATA_DIR, filename)

avail = available_for_subject(selected_subj)
if (session_str, task_str) not in avail:
    req_line = f"You asked for: {pretty_condition_suffix('SD' if session_str=='ses-2' else 'NS')} ({session_str}) • {'Eyes Open' if task_str=='eyesopen' else 'Eyes Closed'}"
    if avail:
        lines = []
        for (ses, task) in sorted(avail.keys()):
            cond = "NS" if ses == "ses-1" else "SD"
            cond_name = pretty_condition_suffix(cond)
            task_name = "Eyes Open" if task == "eyesopen" else "Eyes Closed"
            lines.append(f"- {cond_name} ({ses}) • {task_name}")
        avail_block = "\n".join(lines)
    else:
        avail_block = "- None"
    st.info(
        "This participant does not have that recording.\n\n"
        f"• {req_line}\n"
        f"• Available for this participant:\n{avail_block}\n\n"
        "Try switching Condition or Task above."
    )
    st.stop()

if not os.path.exists(file_path):
    with st.spinner(f"Fetching {filename}..."):
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO, filename=filename, repo_type="dataset",
                local_dir=DATA_DIR, local_dir_use_symlinks=False
            )
            file_path = downloaded
        except Exception as e:
            st.error(
                "Could not download the EEG file for this selection. "
                "Try another selection or check your network.\n\n"
                f"Details: {e}"
            )
            st.stop()

# Load data
try:
    df = pd.read_csv(file_path)
except Exception as e:
    st.error(
        "Found the file but could not read it. The file may be corrupted. Try another selection.\n\n"
        f"Details: {e}"
    )
    st.stop()

time_col = "Time"
if time_col not in df.columns:
    st.error("This file is missing the 'Time' column expected by the viewer. Try another recording.")
    st.stop()

# Confirm at least one nonempty EEG column exists
all_cols = [c for c in df.columns if c != time_col]
nonempty_channels_in_file = [c for c in all_cols if df[c].notna().any()]
if not nonempty_channels_in_file:
    st.info("This recording does not contain usable EEG channels. Try a different task or condition for this participant.")
    st.stop()

# =============================================================================
# Channel picker (now that we know the file's channels)
# =============================================================================
have_positions = bool(USE_COORDS or SVG_POS)

# Determine regions to use for "allowed" list
if st.session_state.selected_regions or st.session_state.region_mode == "All":
    regions_for_allowed = (
        st.session_state.selected_regions
        if st.session_state.selected_regions
        else set(REGION_MAP.keys())  # true ALL when nothing toggled
    )
    wanted = set()
    for r in regions_for_allowed:
        wanted.update(REGION_MAP.get(r, []))
else:
    regions_for_allowed = set()
    wanted = set()

# Allowed = in wanted (or empty if no regions) ∩ has coordinates ∩ present (non-empty) in this file
if wanted:
    allowed = sorted([
        c for c in wanted
        if (get_xy(c) is not None if have_positions else True) and (c in nonempty_channels_in_file)
    ])
else:
    allowed = []  # no regions picked -> no suggestions

# Sanitize current selection so defaults are always valid
current = st.session_state.get("channel_sel", [])
sanitized = [c for c in current if c in allowed]
if sanitized != current:
    st.session_state.channel_sel = sanitized  # drop anything not valid now

# --- Instruction ABOVE the channel picker (bold/italic + slight color)
st.markdown(
    '<div style="margin:0.25rem 0 0.25rem 0;">'
    '<span style="font-weight:700; font-style:italic; color:#3B82F6;">'
    'Pick one or more channels from the list above to view the signal.'
    '</span>'
    '</div>',
    unsafe_allow_html=True
)

# Channel picker UI
selected_channels = st.multiselect(
    "Channels:",
    allowed,
    default=st.session_state.channel_sel,
    key="channel_sel",
    help="Pick one or more channels from the selected region(s)."
)

# Bulk actions for channels
bulk1, bulk2 = st.columns(2)
bulk1.button(
    "Add all channels in these regions",
    use_container_width=True,
    disabled=not allowed,
    on_click=lambda: st.session_state.update({"channel_sel": list(allowed)})
)
bulk2.button(
    "Clear channels",
    use_container_width=True,
    disabled=not st.session_state.channel_sel,
    on_click=lambda: st.session_state.update({"channel_sel": []})
)

# Channel glossary above the map, only for selected channels
if selected_channels:
    with st.expander("What do the selected channels mean?", expanded=True):
        for ch in selected_channels:
            plain, tech = channel_explainer(ch)
            st.markdown(f"- **{ch}**  \n  {plain}  \n  *{tech}*")
else:
    st.caption("Select channels to see a short glossary.")

# =============================================================================
# Brain map
# =============================================================================
# Regions to draw on the map:
if st.session_state.region_mode == "All" and not st.session_state.selected_regions:
    regions_to_draw = set(REGION_MAP.keys())
else:
    regions_to_draw = st.session_state.selected_regions

show_map = st.checkbox("Show brain map", value=True)
if show_map:
    st.plotly_chart(
        build_brain_map(selected_channels, regions_to_draw, SVG_BYTES),
        use_container_width=True
    )
    st.caption("Top view. Forehead at the top, back of head at the bottom. Channel dot positions are approximate.")

# =============================================================================
# Signal plot
# =============================================================================
if selected_channels:
    fig = go.Figure()
    for ch in selected_channels:
        if ch in df.columns:
            fig.add_trace(go.Scatter(x=df[time_col], y=df[ch], mode="lines", name=ch))
    fig.update_layout(
        title=pretty_title(subj_str, selected_cond, selected_task),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (μV)",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

