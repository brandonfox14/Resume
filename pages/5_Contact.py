import streamlit as st
from PIL import Image

st.set_page_config(page_title="Contact", layout="wide")

# --- HEADER / INTRO ---
col1, col2 = st.columns([1, 3])

with col1:
    profile_pic = Image.open("assets/Head_shot.jpeg")
    st.image(profile_pic, width=180)

with col2:
    st.title("Brandon Fox")
    st.subheader("Contact Information")
    st.write(
        "I am always open to discussing new opportunities, collaborations, "
        "or sharing more about my projects. Please feel free to connect with me."
    )

    st.markdown(
        """
        Madison, WI  
        [Email](mailto:brandonfox14@icloud.com) | 
        [LinkedIn](https://linkedin.com/in/brandon-fox-6446261a8)
        """
    )

