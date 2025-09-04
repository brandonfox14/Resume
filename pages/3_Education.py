import streamlit as st
import os

st.set_page_config(page_title="Education", layout="wide")

# --- Helper function to safely load images ---
def safe_image(path, **kwargs):
    if os.path.exists(path):
        st.image(path, **kwargs)
    else:
        st.warning(f"Missing image: {path}")

# --- PAGE HEADER ---
st.title("Education")

st.write(
    "My academic background has provided a strong foundation in data science, "
    "economics, and statistical modeling, equipping me to approach analytical problems "
    "with both technical and applied perspectives."
)

st.divider()

# --- University of Michigan ---
col1, col2 = st.columns([1, 5])
with col1:
    safe_image("assets/logos/University_of_Michigan_Logo.png", width=120)
with col2:
    st.subheader("Master of Science: Applied Data Science")
    st.caption("University of Michigan – Ann Arbor, MI")

st.divider()

# --- Illinois State University ---
col1, col2 = st.columns([1, 5])
with col1:
    safe_image("assets/logos/Illinois_State.png", width=120)
with col2:
    st.subheader("Bachelor of Science: Economics")
    st.caption("Illinois State University – Normal, IL")

