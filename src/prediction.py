import joblib
import pandas as pd

def predict(input_dict):
    model = joblib.load("model.pkl")

    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction[0]),
        "dropout_probability": float(probability)
    }
