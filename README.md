# Student Burnout & Dropout Prediction System

## Problem Statement
Early detection of student burnout and dropout risk using behavioral analytics.

## Dataset
Synthetic dataset (10,000 students) generated using behavioral segmentation:
- LMS activity
- Attendance
- Assignment delay
- Sentiment analysis
- Activity irregularity

## Features Engineered
- Engagement score
- Delay ratio
- Negative sentiment flag
- Irregular behavior flag
- Risk score

## Models Used
- Logistic Regression (Dropout Prediction)
- Random Forest (Burnout Classification)

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Dashboard
Built using Power BI to visualize:
- Risk distribution
- Burnout vs Dropout
- Feature impact
- Dropout probability histogram
- LMS vs Burnout heatmap

## How to Run
1. Run data_generation.py
2. Run feature_engineering.py
3. Run model_training.py
