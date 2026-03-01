{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
from sklearn.model_selection import train_test_split\
from sklearn.linear_model import LogisticRegression\
from sklearn.ensemble import RandomForestClassifier\
from sklearn.metrics import classification_report, roc_auc_score\
import joblib\
\
def train_model():\
    df = pd.read_csv("data/engineered_student_data.csv")\
\
    features = [\
        "engagement_score",\
        "delay_ratio",\
        "negative_sentiment_flag",\
        "irregular_behavior_flag"\
    ]\
\
    X = df[features]\
    y = df["dropout_label"]\
\
    X_train, X_test, y_train, y_test = train_test_split(\
        X, y, test_size=0.2, random_state=42\
    )\
\
    model = LogisticRegression()\
    model.fit(X_train, y_train)\
\
    y_pred = model.predict(X_test)\
    y_prob = model.predict_proba(X_test)[:, 1]\
\
    print(classification_report(y_test, y_pred))\
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))\
\
    joblib.dump(model, "model.pkl")\
    print("Model Saved!")\
\
if __name__ == "__main__":\
    train_model()}