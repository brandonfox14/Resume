import streamlit as st
import os

st.set_page_config(page_title="Experience", layout="wide")

# --- Helper function to safely load images ---
def safe_image(path, **kwargs):
    if os.path.exists(path):
        st.image(path, **kwargs)
    else:
        st.warning(f"Missing image: {path}")

# --- PAGE HEADER ---
st.title("Work Experience")

st.write(
    "A summary of my professional experience across data science, research, and analytics. "
    "Each role reflects my ability to apply statistical methods, predictive modeling, and "
    "data storytelling to solve real-world problems."
)

st.divider()

# --- March Metrics ---
col1, col2 = st.columns([1, 5])
with col1:
    safe_image("assets/logos/FullLogo.png", width=120)
with col2:
    st.subheader("Data Scientist")
    st.caption("March Metrics – Remote | 08/2020 to Present")
    st.markdown(
        """
        - Designed and deployed an automated data pipeline to collect, clean, and store up-to-date player and team statistics daily, ensuring real-time data availability for modeling and analysis.  
        - Conduct ongoing trend analysis of team performance and individual player metrics, leveraging advanced statistical techniques and time series data to surface meaningful patterns and behavioral shifts.  
        - Built predictive models using ensemble methods and regression-based algorithms to forecast game outcomes and player performance, supporting scenario planning and probability-based strategy.  
        - Developed interactive visualizations and dashboards to highlight key trends, edge cases, and high-value opportunities within upcoming games—enabling clients to identify strategic advantages in a regulated wagering environment.  
        """
    )

st.divider()

# --- Illinois State University ---
col1, col2 = st.columns([1, 5])
with col1:
    safe_image("assets/logos/Illinois_State.png", width=120)
with col2:
    st.subheader("Graduate Assistant – Data and Statistical Researcher")
    st.caption("Illinois State University – Normal, IL | 08/2023 to 05/2024")
    st.markdown(
        """
        - Performed EDA to gain insights into the data, including summary statistics, data visualization, and correlation analysis, to understand patterns, trends, and relationships within the data.  
        - Developed and trained machine learning models using algorithms such as linear regression, logistic regression, decision trees, random forests, neural networks, and ensemble methods.  
        - Supported faculty with their data analysis, statistical consulting, and data visualization needs related to their research interests.  
        - Performed logistic regression analysis using students’ home address data to evaluate the relationship between distance from home and retention rates, revealing location-based trends that informed strategic retention initiatives.  
        """
    )

st.divider()

# --- Sushi Primos ---
col1, col2 = st.columns([1, 5])
with col1:
    safe_image("assets/logos/SUSHI_PRIMOS_LOGO.png", width=120)
with col2:
    st.subheader("Market Analyst")
    st.caption("Sushi Primos – Normal, IL | 09/2021 to 09/2023")
    st.markdown(
        """
        - Collected and integrated external datasets (e.g., traffic volume, residential density, foot traffic) with internal sales data to assess market potential for expansion.  
        - Developed visualizations and simplified reporting tools to communicate findings to non-technical stakeholders.  
        - Conducted pricing analysis based on historical sales data to refine product pricing strategy and optimize profitability across high-performing menu items.  
        - Represented the business in meetings with local government officials, translating regulatory requirements and incentives into clear terms for the owner.  
        """
    )

st.divider()

# --- UPS ---
col1, col2 = st.columns([1, 5])
with col1:
    safe_image("assets/logos/UPS.png", width=120)
with col2:
    st.subheader("Supervisor")
    st.caption("UPS – Middleton, WI | 06/2020 to 08/2021")
    st.markdown(
        """
        - Supervised daily operations, ensuring efficiency and accuracy of logistics processes.  
        - Managed and trained staff while optimizing scheduling and workflow.  
        - Applied data-driven approaches to improve package routing and reduce errors.  
        """
    )

