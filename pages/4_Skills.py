import streamlit as st

st.set_page_config(page_title="Skills", layout="wide")

# --- PAGE HEADER ---
st.title("Skills")

st.write(
    "A blend of technical expertise and applied problem-solving skills in data science, "
    "analytics, and communication."
)

st.divider()

# --- Technical Skills ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data & Analysis")
    st.write(
        "Data Analysis, Data Visualization, Data Mining, "
        "Data Modeling, Data Management"
    )

    st.subheader("Programming")
    st.write("Python, R, SQL")

    st.subheader("Databases")
    st.write("MySQL, PostgreSQL, SQL Server")

with col2:
    st.subheader("Tools")
    st.write("Tableau, Power BI, Excel")

    st.subheader("Statistical Analysis")
    st.write("Regression, Predictive Modeling, Machine Learning")

    st.subheader("Business Intelligence & Reporting")
    st.write("Dashboard Development, Data Warehousing")

st.divider()

# --- Soft Skills ---
st.subheader("Soft Skills")
st.write(
    "Communication, Problem-Solving, Critical Thinking, "
    "Collaboration, Organization"
)

