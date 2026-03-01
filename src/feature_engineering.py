{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
\
def feature_engineering():\
    df = pd.read_csv("data/synthetic_student_data.csv")\
\
    df["engagement_score"] = (\
        0.5 * (df["lms_logins_per_week"] / 20) +\
        0.5 * (df["attendance_percentage"] / 100)\
    )\
\
    df["delay_ratio"] = df["avg_assignment_delay_days"] / 15\
\
    df["negative_sentiment_flag"] = df["feedback_sentiment_score"].apply(\
        lambda x: 1 if x < -0.3 else 0\
    )\
\
    df["irregular_behavior_flag"] = df["activity_irregularity_score"].apply(\
        lambda x: 1 if x > 0.7 else 0\
    )\
\
    low_lms = 1 - (df["lms_logins_per_week"] / 20)\
    delay_score = df["avg_assignment_delay_days"] / 15\
    low_attendance = 1 - (df["attendance_percentage"] / 100)\
    negative_sentiment = np.clip(-df["feedback_sentiment_score"], 0, 1)\
\
    df["risk_score"] = (\
        0.25 * low_lms +\
        0.25 * low_attendance +\
        0.20 * delay_score +\
        0.15 * negative_sentiment +\
        0.15 * df["activity_irregularity_score"]\
    ) * 100\
\
    df["burnout_label"] = np.where(df["risk_score"] > 70, 1, 0)\
    df["dropout_probability"] = 1 / (1 + np.exp(-(df["risk_score"] - 50) / 10))\
    df["dropout_label"] = np.where(df["dropout_probability"] > 0.6, 1, 0)\
\
    df.to_csv("data/engineered_student_data.csv", index=False)\
    print("Feature Engineering Completed!")\
\
if __name__ == "__main__":\
    feature_engineering()}