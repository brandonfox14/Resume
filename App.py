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
    profile_pic = Image.open("assets/Head_shot.jpeg")
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
        Madison, WI  
        [Email](brandonfox14@icloud.com) | 
        [LinkedIn](https://linkedin.com/in/brandon-fox-6446261a8)
        """
    )

st.divider()


