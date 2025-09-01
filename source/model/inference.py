import joblib
import pandas as pd
# Load pipeline 
loaded_model = joblib.load('gb_tuned.joblib') 
# Collect input from the user 
user_input = { 'Age': int(input("Enter Age: ")), 
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
               'School_Grade': input("Enter School Grade (7th-12th): ") } 

# Convert to DataFrame 
new_data = pd.DataFrame([user_input]) 

# Predict
y_pred_new = loaded_model.predict(new_data) 
y_pred_proba_new = loaded_model.predict_proba(new_data) 
print("Predicted Class:", y_pred_new) 
print("The addiction level is: ", y_pred_new) 
print("Prediction Probabilities:", y_pred_proba_new)