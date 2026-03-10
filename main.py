import pickle
import pandas as pd

# load model and scaler
model = pickle.load(open("diabetes_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

print("----- Diabetes Prediction -----")

gender = input("Gender (Male/Female): ")
age = float(input("Age: "))
hypertension = int(input("Hypertension (0/1): "))
heart_disease = int(input("Heart Disease (0/1): "))
smoking_history = input("Smoking History: ")
bmi = float(input("BMI: "))
hba1c = float(input("HbA1c Level: "))
glucose = float(input("Blood Glucose Level: "))

# ===============================
# FEATURE ENGINEERING (same as training)
# ===============================

high_hba1c = 1 if hba1c >= 6.5 else 0
high_glucose = 1 if glucose >= 200 else 0

prediabetes_hba1c = 1 if 5.7 <= hba1c < 6.5 else 0
prediabetes_glucose = 1 if 126 <= glucose < 200 else 0

obese = 1 if bmi >= 30 else 0


# create dataframe
data = pd.DataFrame({
    "gender":[gender],
    "age":[age],
    "hypertension":[hypertension],
    "heart_disease":[heart_disease],
    "smoking_history":[smoking_history],
    "bmi":[bmi],
    "HbA1c_level":[hba1c],
    "blood_glucose_level":[glucose],
    "high_hba1c":[high_hba1c],
    "high_glucose":[high_glucose],
    "prediabetes_hba1c":[prediabetes_hba1c],
    "prediabetes_glucose":[prediabetes_glucose],
    "obese":[obese]
})

# apply encoding
data = pd.get_dummies(data, drop_first=True)

# expected columns from training
expected_columns = [
'age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level',
'high_hba1c','high_glucose','prediabetes_hba1c','prediabetes_glucose','obese',
'gender_Male','gender_Other',
'smoking_history_current','smoking_history_ever',
'smoking_history_former','smoking_history_never','smoking_history_not current'
]

# add missing columns
for col in expected_columns:
    if col not in data.columns:
        data[col] = 0

# reorder columns
data = data[expected_columns]

# scale input
data_scaled = scaler.transform(data)

# prediction probability
prob = model.predict_proba(data_scaled)[0][1]

print("Diabetes Risk:", round(prob*100,2), "%")

if prob > 0.35:
    print("Person is LIKELY to have Diabetes ⚠️")
else:
    print("Person is NOT likely to have Diabetes ")



    """
Load the saved model (diabetes_model.pkl)
Take input from the user
Convert input into the correct format
Predict using the model
Show the result
"""