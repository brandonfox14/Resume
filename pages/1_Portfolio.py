import streamlit as st
import os

st.set_page_config(page_title="Portfolio", layout="wide")

# --- Helper function to safely load images ---
def safe_image(path, **kwargs):
    if os.path.exists(path):
        st.image(path, **kwargs)
    else:
        st.warning(f"Missing image: {path}")

# --- PAGE HEADER ---
st.title("Portfolio")
st.write(
    "This portfolio highlights recent professional and educational projects that "
    "demonstrate my ability to apply data science, predictive modeling, and interactive "
    "application design. Each project represents a step in combining technical expertise "
    "with practical, real-world impact."
)

st.divider()

# --- Michigan Capstone Project ---
st.header("Michigan Capstone Project")
st.write(
    "This application was developed as part of my Master's program at the University of Michigan. "
    "It demonstrates applied data science workflows, predictive modeling, and interactive "
    "visualizations. The project serves as a capstone deliverable as well as a professional "
    "example of building and deploying analytics-driven applications."
)

cap_col1, cap_col2 = st.columns(2)
with cap_col1:
    safe_image("assets/Mich1.png", use_container_width=True)
with cap_col2:
    safe_image("assets/Mich2.png", use_container_width=True)

st.markdown("[Explore the full app](https://michigancapstone-forresume.streamlit.app)")

st.divider()

# --- March Metrics Project ---
st.header("March Metrics")
st.write(
    "March Metrics is an NCAA basketball analytics platform designed to provide advanced insights "
    "into team and player performance. The app integrates custom efficiency statistics, predictive "
    "models, and interactive dashboards to explore matchups and outcomes. This project demonstrates "
    "how analytics can be made accessible to coaches, analysts, and fans through clean design and "
    "data storytelling."
)

mm_col1, mm_col2, mm_col3 = st.columns(3)
with mm_col1:
    safe_image("assets/March_Metrics1.png", use_container_width=True)
with mm_col2:
    safe_image("assets/March_Metrics2.png", use_container_width=True)
with mm_col3:
    safe_image("assets/March_Metrics3.png", use_container_width=True)

st.markdown("[Explore the full app](https://march-metrics-resume-app.streamlit.app)")

st.divider()
