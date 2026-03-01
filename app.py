import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Burnout Prediction Dashboard", layout="wide")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("data/synthetic_student_data.csv")

df = load_data()

# =========================
# Auto Fix Missing Columns
# =========================

# Create risk_category if missing
if "risk_category" not in df.columns and "risk_score" in df.columns:
    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[-1, 33, 66, 100],
        labels=["Low", "Medium", "High"]
    )

# Create burnout_label if missing
if "burnout_label" not in df.columns and "risk_score" in df.columns:
    df["burnout_label"] = (df["risk_score"] > 60).astype(int)

# Create dropout_probability if missing
if "dropout_probability" not in df.columns and "risk_score" in df.columns:
    df["dropout_probability"] = df["risk_score"] / 100

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

with col5:
    st.subheader("Risk Distribution")
    fig1, ax1 = plt.subplots()
    df["risk_category"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax1
    )
    ax1.set_ylabel("")
    st.pyplot(fig1)

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

with col7:
    st.subheader("Feature Impact Overview")

    features = [
        "engagement_score",
        "delay_ratio",
        "negative_sentiment_flag",
        "irregular_behavior_flag"
    ]

    existing_features = [f for f in features if f in df.columns]

    if existing_features:
        feature_means = df[existing_features].mean()
        fig3, ax3 = plt.subplots()
        feature_means.plot(kind="bar", ax=ax3)
        ax3.set_ylabel("Average Value")
        st.pyplot(fig3)
    else:
        st.warning("Feature columns not found in dataset.")

with col8:
    st.subheader("Attendance vs Risk Score")

    if "attendance_percentage" in df.columns:
        fig4, ax4 = plt.subplots()
        ax4.scatter(
            df["attendance_percentage"],
            df["risk_score"],
            alpha=0.5
        )
        ax4.set_xlabel("Attendance Percentage")
        ax4.set_ylabel("Risk Score")
        st.pyplot(fig4)
    else:
        st.warning("Attendance column not found.")

st.markdown("---")

# =========================
# LMS vs Burnout Heatmap
# =========================
st.subheader("LMS Logins vs Burnout Heatmap")

if "lms_logins_per_week" in df.columns:
    heatmap_data = pd.crosstab(
        pd.qcut(df["lms_logins_per_week"], 5, duplicates="drop"),
        df["burnout_label"]
    )

    fig5, ax5 = plt.subplots()
    im = ax5.imshow(heatmap_data, aspect="auto")

    ax5.set_xticks(range(len(heatmap_data.columns)))
    ax5.set_xticklabels(heatmap_data.columns)

    ax5.set_yticks(range(len(heatmap_data.index)))
    ax5.set_yticklabels(heatmap_data.index)

    plt.colorbar(im)
    st.pyplot(fig5)
else:
    st.warning("LMS login column not found.")

st.markdown("---")

st.success("Dashboard Loaded Successfully 🚀")
