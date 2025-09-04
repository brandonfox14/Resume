import streamlit as st

st.set_page_config(page_title="Portfolio", page_icon="📂", layout="wide")

st.title("📂 Portfolio")
st.write("A collection of professional and educational projects I've built and shared. Each app demonstrates applied data science, predictive modeling, and interactive design using Streamlit.")

st.divider()

# --- Michigan Capstone Project ---
st.header("🎓 Michigan Capstone Project")
st.write(
    """
    This is a **professional and educational Streamlit application** created as part of my 
    Master’s program at the University of Michigan.  

    The app demonstrates advanced data science workflows, predictive modeling, and interactive 
    visualization techniques. It serves as both a **capstone deliverable** and a **professional portfolio piece** 
    to showcase my ability to build and deploy applied analytics solutions.  
    """
)

cap_col1, cap_col2 = st.columns(2)
with cap_col1:
    st.image("assets/screenshots/Mich1.png", use_container_width=True)
    st.image("assets/screenshots/Mich2.png", use_container_width=True)
with cap_col2:
    st.markdown("[🌐 Explore the Full App](https://michigancapstone-forresume.streamlit.app)")

st.divider()

# --- March Metrics Project ---
st.header("🏀 March Metrics")
st.write(
    """
    **March Metrics** is an NCAA basketball analytics platform built in Streamlit.  

    The app provides:
    - Custom statistics and advanced efficiency metrics  
    - Game insights with predictive modeling  
    - Interactive dashboards for exploring teams and matchups  

    This project highlights my ability to connect **data engineering, 
    statistical modeling, and intuitive UI design** in a way that makes analytics 
    approachable for coaches, analysts, and fans.  
    """
)

mm_col1, mm_col2, mm_col3 = st.columns(3)
with mm_col1:
    st.image("assets/screenshots/March_Metrics1.png", use_container_width=True)
with mm_col2:
    st.image("assets/screenshots/March_Metrics2.png", use_container_width=True)
with mm_col3:
    st.image("assets/screenshots/March_Metrics3.png", use_container_width=True)

st.markdown("[🌐 Explore the Full App](https://march-metrics-resume-app.streamlit.app)")

st.divider()

