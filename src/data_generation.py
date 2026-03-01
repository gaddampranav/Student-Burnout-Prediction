import numpy as np
import pandas as pd

def generate_synthetic_data(n=10000, seed=42):
    np.random.seed(seed)

    student_id = np.arange(1, n+1)

    student_type = np.random.choice(
        ["normal", "medium_risk", "high_risk"],
        size=n,
        p=[0.6, 0.25, 0.15]
    )

    lms_logins = []
    assignment_delay = []
    attendance = []
    sentiment = []
    irregularity = []

    for s in student_type:
        if s == "normal":
            lms_logins.append(np.random.randint(10, 21))
            assignment_delay.append(np.random.randint(0, 5))
            attendance.append(np.random.uniform(75, 100))
            sentiment.append(np.random.uniform(0, 1))
            irregularity.append(np.random.uniform(0, 0.4))
        elif s == "medium_risk":
            lms_logins.append(np.random.randint(5, 12))
            assignment_delay.append(np.random.randint(3, 10))
            attendance.append(np.random.uniform(60, 80))
            sentiment.append(np.random.uniform(-0.5, 0.3))
            irregularity.append(np.random.uniform(0.3, 0.7))
        else:
            lms_logins.append(np.random.randint(0, 6))
            assignment_delay.append(np.random.randint(7, 16))
            attendance.append(np.random.uniform(40, 65))
            sentiment.append(np.random.uniform(-1, 0))
            irregularity.append(np.random.uniform(0.6, 1))

    df = pd.DataFrame({
        "student_id": student_id,
        "student_type": student_type,
        "lms_logins_per_week": lms_logins,
        "avg_assignment_delay_days": assignment_delay,
        "attendance_percentage": attendance,
        "feedback_sentiment_score": sentiment,
        "activity_irregularity_score": irregularity
    })

    df.to_csv("data/synthetic_student_data.csv", index=False)
    print("Dataset Generated Successfully!")

if __name__ == "__main__":
    generate_synthetic_data()
