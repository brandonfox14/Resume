import streamlit as st
from PIL import Image

# --- CONFIG ---
st.set_page_config(
    page_title="Brandon Fox | Portfolio",
    page_icon="📊",
    layout="wide"
)

# --- HEADER / INTRO ---
col1, col2 = st.columns([1, 3])

with col1:
    profile_pic = Image.open("assets/Head_shot.jpg")
    st.image(profile_pic, width=180)

with col2:
    st.title("Brandon Fox")
    st.subheader("Data Scientist | Sports Analytics | Coach")
    st.write(
        "Data-driven sports analyst with expertise in predictive modeling, "
        "statistical analysis, and data visualization. Passionate about leveraging "
        "data science to optimize performance, uncover insights, and drive "
        "strategic decision-making in sports and beyond."
    )

    st.markdown(
        """
        📍 Madison, WI  
        ✉️ [Email](mailto:your_email@example.com) | 
        💼 [LinkedIn](https://linkedin.com/in/yourusername) | 
        💻 [GitHub](https://github.com/yourusername)
        """
    )

st.divider()

# --- QUICK NAVIGATION ---
st.write("### 🔎 Explore My Portfolio")
st.write("Use the sidebar to navigate through my portfolio, experience, skills, and contact information.")

st.divider()

# --- FEATURED PROJECTS (optional quick teaser) ---
st.write("### 🚀 Featured Projects")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**[March Metrics](https://your-streamlit-app1.streamlit.app)**")
    st.caption("NCAA basketball analytics app with custom metrics.")
    st.image("assets/screenshots/march_metrics.png")

with col2:
    st.markdown("**[NASCAR Predictor](https://your-streamlit-app2.streamlit.app)**")
    st.caption("Predictive model for NASCAR fantasy points.")
    st.image("assets/screenshots/nascar_predictor.png")

st.divider()

st.write("⬅️ Navigate with the sidebar to see more details.")
