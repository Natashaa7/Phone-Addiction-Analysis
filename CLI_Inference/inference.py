import joblib
import pandas as pd
from pathlib import Path

class AddictionPredictor:
    def __init__(self, model_path: str = "../model/gb_tuned.joblib"):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        print("✅ Model loaded successfully!")

    def predict(self, user_input: dict):
        """Predict class and probabilities for a single user input."""
        # Wrap dict in a list and convert to DataFrame
        df = pd.DataFrame([user_input])
        y_pred = self.model.predict(df)
        y_proba = self.model.predict_proba(df)
        return y_pred, y_proba

if __name__ == "__main__":
    predictor = AddictionPredictor()

    user_input = {
        'Age': int(input("Enter Age: ")),
        'Daily_Usage_Hours': float(input("Enter Daily Usage Hours: ")),
        'Sleep_Hours': float(input("Enter Sleep Hours: ")),
        'Academic_Performance': int(input("Enter Academic Performance (1-100): ")),
        'Social_Interactions': int(input("Enter Social Interactions (scale 1-10): ")),
        'Exercise_Hours': float(input("Enter Exercise Hours: ")),
        'Anxiety_Level': int(input("Enter Anxiety Level (scale 1-10): ")),
        'Depression_Level': int(input("Enter Depression Level (scale 1-10): ")),
        'Self_Esteem': int(input("Enter Self Esteem Level (scale 1-10): ")),
        'Phone_Checks_Per_Day': int(input("Enter Phone Checks Per Day: ")),
        'Apps_Used_Daily': int(input("Enter Number of Apps Used Daily: ")),
        'Time_on_Social_Media': float(input("Enter Time on Social Media (hours): ")),
        'Time_on_Gaming': float(input("Enter Time on Gaming (hours): ")),
        'Time_on_Education': float(input("Enter Time on Education (hours): ")),
        'Family_Communication': int(input("Enter Family Communication (scale 1-10): ")),
        'Weekend_Usage_Hours': float(input("Enter Weekend Usage Hours: ")),
        'Gender': input("Enter Gender (Male/Female/Other): "),
        'Phone_Usage_Purpose': input("Enter Phone Usage Purpose (Browsing/Education/Social Media/Gaming/Other): "),
        'School_Grade': input("Enter School Grade (7th-12th): ")
    }

    y_pred, y_proba = predictor.predict(user_input)

    print("\n📊 Results:")
    print("Predicted Class:", y_pred[0])
    print("The addiction level is:", y_pred[0])
    print("Prediction Probabilities:", y_proba[0])
