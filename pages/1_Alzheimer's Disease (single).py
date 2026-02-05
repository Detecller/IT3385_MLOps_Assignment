import streamlit as st
import pandas as pd
from mlops_assignment.predict_alzheimer import predict

st.title("🧠 Alzheimer's Disease Prediction (Single)")

# User Inputs
st.header("Demographic Details")

age = st.slider("Age", 55, 95, 75)

gender = st.selectbox(
    "Gender",
    [0, 1],
    format_func=lambda x: "Male" if x == 0 else "Female"
)

ethnicity = st.selectbox(
    "Ethnicity",
    [0,1,2,3],
    format_func=lambda x: ["Caucasian","African American","Asian","Other"][x]
)

education = st.selectbox(
    "Education Level",
    [0,1,2,3],
    format_func=lambda x: ["None","High School","Bachelor's","Higher"][x]
)


# ======================
# Lifestyle
# ======================

st.header("Lifestyle")

bmi = st.slider("BMI", 13.0, 44.0, 27.5)
smoking = st.selectbox("Smoking", [0,1])

alcohol = st.slider(
    "Alcohol Consumption (units/week)",
    0.0, 22.0, 10.0
)

physical = st.slider(
    "Physical Activity (hrs/week)",
    0.0, 11.0, 5.0
)

diet = st.slider("Diet Quality", 0.0, 10.0, 5.0)
sleep = st.slider("Sleep Quality", 3.0, 10.0, 7.0)


# ======================
# Medical History
# ======================

st.header("Medical History")

family = st.selectbox("Family History Alzheimer's", [0,1])
cardio = st.selectbox("Cardiovascular Disease", [0,1])
diabetes = st.selectbox("Diabetes", [0,1])
depression = st.selectbox("Depression", [0,1])
headinjury = st.selectbox("Head Injury", [0,1])
hypertension = st.selectbox("Hypertension", [0,1])


# ======================
# Clinical Measurements
# ======================

st.header("Clinical Measurements")

sbp = st.slider("Systolic BP", 85, 190, 135)
dbp = st.slider("Diastolic BP", 55, 125, 90)

chol_total = st.slider("Total Cholesterol", 130, 330, 225)
chol_ldl = st.slider("LDL", 40, 220, 125)
chol_hdl = st.slider("HDL", 15, 110, 60)
trig = st.slider("Triglycerides", 40, 440, 230)


# ======================
# Cognitive & Functional
# ======================

st.header("Cognitive & Functional")

mmse = st.slider("MMSE", 0, 30, 15)
functional = st.slider("Functional Assessment", 0, 10, 5)
memory = st.selectbox("Memory Complaints", [0,1])
behavior = st.selectbox("Behavioral Problems", [0,1])
adl = st.slider("ADL", 0, 10, 5)


# ======================
# Symptoms
# ======================

st.header("Symptoms")

confusion = st.selectbox("Confusion", [0,1])
disorientation = st.selectbox("Disorientation", [0,1])
personality = st.selectbox("Personality Changes", [0,1])
tasks = st.selectbox("Difficulty Completing Tasks", [0,1])
forgetful = st.selectbox("Forgetfulness", [0,1])

# Build dataframe — column names MUST match training data
input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Ethnicity": ethnicity,
    "EducationLevel": education,
    "BMI": bmi,
    "Smoking": smoking,
    "AlcoholConsumption": alcohol,
    "PhysicalActivity": physical,
    "DietQuality": diet,
    "SleepQuality": sleep,
    "FamilyHistoryAlzheimers": family,
    "CardiovascularDisease": cardio,
    "Diabetes": diabetes,
    "Depression": depression,
    "HeadInjury": headinjury,
    "Hypertension": hypertension,
    "SystolicBP": sbp,
    "DiastolicBP": dbp,
    "CholesterolTotal": chol_total,
    "CholesterolLDL": chol_ldl,
    "CholesterolHDL": chol_hdl,
    "CholesterolTriglycerides": trig,
    "MMSE": mmse,
    "FunctionalAssessment": functional,
    "MemoryComplaints": memory,
    "BehavioralProblems": behavior,
    "ADL": adl,
    "Confusion": confusion,
    "Disorientation": disorientation,
    "PersonalityChanges": personality,
    "DifficultyCompletingTasks": tasks,
    "Forgetfulness": forgetful
}])

if st.button("Predict"):
    pred, prob = predict(input_df)

    label_map = {
        0: "Unlikely to have Alzheimer’s disease",
        1: "Likely to have Alzheimer’s disease"
    }

    if pred.iloc[0] == 0:
        st.success(f"{label_map[pred.iloc[0]]} (Confidence: {prob.iloc[0]:.2%})")
    else:
        st.error(f"{label_map[pred.iloc[0]]} (Confidence: {prob.iloc[0]:.2%})")