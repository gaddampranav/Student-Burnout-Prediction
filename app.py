import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Burnout Prediction Dashboard", layout="wide")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/synthetic_student_data.csv")
    return df

df = load_data()

# =========================
# Title
# =========================
st.title("🎓 Student Burnout & Risk Analytics Dashboard")
st.markdown("---")

# =========================
# KPI Section
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_risk = df["risk_score"].mean()
    st.metric("Average Risk Score", f"{avg_risk:.2f}")

with col2:
    high_risk_pct = (df["risk_category"] == "High").mean() * 100
    st.metric("High Risk %", f"{high_risk_pct:.1f}%")

with col3:
    burnout_pct = df["burnout_label"].mean() * 100
    st.metric("Burnout Rate %", f"{burnout_pct:.1f}%")

with col4:
    dropout_pct = df["dropout_probability"].mean() * 100
    st.metric("Predicted Dropout %", f"{dropout_pct:.1f}%")

st.markdown("---")

# =========================
# Row 1: Pie + Histogram
# =========================
col5, col6 = st.columns(2)

# Pie Chart
with col5:
    st.subheader("Risk Distribution")
    fig1, ax1 = plt.subplots()
    df["risk_category"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax1
    )
    ax1.set_ylabel("")
    st.pyplot(fig1)

# Dropout Histogram
with col6:
    st.subheader("Dropout Probability Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df["dropout_probability"], bins=20)
    ax2.set_xlabel("Dropout Probability")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

st.markdown("---")

# =========================
# Row 2: Feature Impact + Scatter
# =========================
col7, col8 = st.columns(2)

# Feature Importance (Average Risk by Feature)
with col7:
    st.subheader("Feature Impact on Risk Score")

    features = [
        "engagement_score",
        "delay_ratio",
        "negative_sentiment_flag",
        "irregular_behavior_flag"
    ]

    feature_means = df[features].mean()

    fig3, ax3 = plt.subplots()
    feature_means.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Average Value")
    st.pyplot(fig3)

# Scatter Plot
with col8:
    st.subheader("Attendance vs Risk Score")

    fig4, ax4 = plt.subplots()
    scatter = ax4.scatter(
        df["attendance_percentage"],
        df["risk_score"],
        alpha=0.5
    )
    ax4.set_xlabel("Attendance Percentage")
    ax4.set_ylabel("Risk Score")
    st.pyplot(fig4)

st.markdown("---")

# =========================
# LMS vs Burnout Heatmap
# =========================
st.subheader("LMS Logins vs Burnout Heatmap")

heatmap_data = pd.crosstab(
    pd.qcut(df["lms_logins_per_week"], 5),
    df["burnout_label"]
)

fig5, ax5 = plt.subplots()
sns.heatmap(heatmap_data, annot=True, fmt="d", ax=ax5)
st.pyplot(fig5)

st.markdown("---")

st.success("Dashboard Loaded Successfully 🚀")
