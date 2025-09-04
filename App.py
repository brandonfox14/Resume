import streamlit as st
from PIL import Image

# --- CONFIG ---
st.set_page_config(
    page_title="Brandon Fox | Resume Portfolio",
    page_icon="📊",
    layout="wide"
)

# --- HEADER / INTRO ---
col1, col2 = st.columns([1, 3])

with col1:
    profile_pic = Image.open("assets/Head_shot.jpeg")
    st.image(profile_pic, width=180)

with col2:
    st.title("Brandon Fox")
    st.subheader("Data Scientist | Sports Analytics | Coach")
    st.write(
        "Welcome to my interactive resume portfolio. This site highlights my professional "
        "experience, technical expertise, and data science projects through a clean and "
        "organized format. Each section is designed to showcase both my technical skills "
        "and ability to apply data-driven solutions to real-world problems."
    )

st.divider()

# --- OVERVIEW OF TABS ---
st.header("Explore the Tabs")

st.subheader("Portfolio")
st.write(
    "Showcases selected projects, including my University of Michigan Capstone project "
    "and March Metrics, an NCAA basketball analytics platform. Each project includes "
    "descriptions, visuals, and links to fully deployed apps."
)

st.subheader("Experience")
st.write(
    "Details my professional and academic work history, including roles as a Data Scientist "
    "with March Metrics, Graduate Assistant at Illinois State University, Market Analyst for "
    "Sushi Primos, and Supervisor at UPS. Each role highlights responsibilities and key contributions."
)

st.subheader("Education")
st.write(
    "Outlines my academic background, including a Master of Science in Applied Data Science "
    "from the University of Michigan and a Bachelor of Science in Economics from Illinois State University."
)

st.subheader("Skills")
st.write(
    "Highlights my technical and soft skills, spanning programming (Python, R, SQL), "
    "data analysis, statistical modeling, visualization, business intelligence tools, "
    "and key collaboration and problem-solving strengths."
)

st.subheader("Contact")
st.write(
    "Provides ways to get in touch, including email and LinkedIn. "
    "This section serves as the easiest way to connect with me directly."
)
