import streamlit as st
import pandas as pd

df = pd.read_csv("../data/engineered_student_data.csv")

st.title("Student Burnout & Dropout Dashboard")

st.metric("Average Risk Score", round(df["risk_score"].mean(), 2))
st.metric("High Risk %", round((df["risk_score"] > 70).mean()*100, 2))

st.bar_chart(df["risk_score"].value_counts())
