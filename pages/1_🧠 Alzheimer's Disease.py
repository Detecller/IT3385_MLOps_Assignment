import streamlit as st
import pandas as pd
from mlops_assignment.predict_alzheimer import predict

st.title("🧠 Alzheimer's Disease Prediction")

# Mode selector
mode = st.radio("Select Prediction Mode:", ["Single Patient", "Batch Prediction"], horizontal=True)

if mode == "Single Patient":
    st.header("Patient Information")
    
    # Demographic Details
    st.subheader("📋 Demographic Details")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age (years)", 60, 90, 75)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
    with col2:
        ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
        education = st.selectbox("Education Level", ["None", "High School", "Bachelor's", "Higher"])
    
    # Lifestyle Factors
    st.subheader("🏃 Lifestyle Factors")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bmi = st.slider("BMI", 15.0, 40.0, 25.0, 0.1, 
                       help="Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30")
        smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        
    with col2:
        alcohol = st.slider("Alcohol Consumption (units/week)", 0, 20, 5,
                           help="Low: 0-7, Moderate: 8-14, High: >14")
        physical_activity = st.slider("Physical Activity (hours/week)", 0, 10, 5,
                                     help="Low: 0-2, Moderate: 3-6, High: >6")
        
    with col3:
        diet_quality = st.slider("Diet Quality (Poor to Good)", 0, 10, 5,
                                help="Poor: 0-3, Fair: 4-6, Good: 7-10")
        sleep_quality = st.slider("Sleep Quality (Poor to Good)", 4, 10, 7,
                                 help="Poor: 4-5, Fair: 6-7, Good: 8-10")
    
    # Medical History
    st.subheader("🏥 Medical History")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        family_history = st.selectbox("Family History of Alzheimer's", ["No", "Yes"])
        cardiovascular = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
        
    with col2:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        depression = st.selectbox("Depression", ["No", "Yes"])
        
    with col3:
        head_injury = st.selectbox("Head Injury History", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    
    # Clinical Measurements
    st.subheader("🩺 Clinical Measurements")
    col1, col2 = st.columns(2)
    
    with col1:
        systolic_bp = st.slider("Systolic BP (mmHg)", 90, 180, 120,
                               help="Normal: <120, Elevated: 120-129, High: ≥130")
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80,
                                help="Normal: <80, Elevated: 80-89, High: ≥90")
        cholesterol_total = st.slider("Total Cholesterol (mg/dL)", 150, 300, 200,
                                     help="Desirable: <200, Borderline: 200-239, High: ≥240")
        
    with col2:
        cholesterol_ldl = st.slider("LDL Cholesterol (mg/dL)", 50, 200, 100,
                                   help="Optimal: <100, Near optimal: 100-129, High: ≥130")
        cholesterol_hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50,
                                   help="Low: <40, Good: 40-59, High: ≥60")
        cholesterol_triglycerides = st.slider("Triglycerides (mg/dL)", 50, 400, 150,
                                             help="Normal: <150, Borderline: 150-199, High: ≥200")
    
    # Cognitive and Functional Assessments
    st.subheader("🧠 Cognitive & Functional Assessments")
    col1, col2 = st.columns(2)
    
    with col1:
        mmse = st.slider("MMSE Score", 0, 30, 24,
                        help="Mini-Mental State Examination. Normal: 24-30, Mild impairment: 18-23, Moderate: 10-17, Severe: <10")
        functional = st.slider("Functional Assessment", 0, 10, 7,
                             help="Higher scores indicate better function. 0=Severely impaired, 10=Independent")
        adl = st.slider("ADL Score", 0, 10, 7,
                       help="Activities of Daily Living. Higher scores indicate better independence. 0=Dependent, 10=Independent")
        
    with col2:
        memory = st.selectbox("Memory Complaints", ["No", "Yes"])
        behavior = st.selectbox("Behavioral Problems", ["No", "Yes"])
    
    # Symptoms
    st.subheader("⚠️ Symptoms")
    col1, col2 = st.columns(2)
    
    with col1:
        confusion = st.selectbox("Confusion", ["No", "Yes"])
        disorientation = st.selectbox("Disorientation", ["No", "Yes"])
        personality_changes = st.selectbox("Personality Changes", ["No", "Yes"])
        
    with col2:
        difficulty_tasks = st.selectbox("Difficulty Completing Tasks", ["No", "Yes"])
        forgetfulness = st.selectbox("Forgetfulness", ["No", "Yes"])
    
    # Convert selections to numeric values
    def binary_to_int(value):
        return 1 if value == "Yes" else 0
    
    ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
    education_map = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}
    
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": 1 if gender == "Female" else 0,
        "Ethnicity": ethnicity_map[ethnicity],
        "EducationLevel": education_map[education],
        "BMI": bmi,
        "Smoking": binary_to_int(smoking),
        "AlcoholConsumption": alcohol,
        "PhysicalActivity": physical_activity,
        "DietQuality": diet_quality,
        "SleepQuality": sleep_quality,
        "FamilyHistoryAlzheimers": binary_to_int(family_history),
        "CardiovascularDisease": binary_to_int(cardiovascular),
        "Diabetes": binary_to_int(diabetes),
        "Depression": binary_to_int(depression),
        "HeadInjury": binary_to_int(head_injury),
        "Hypertension": binary_to_int(hypertension),
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "CholesterolTotal": cholesterol_total,
        "CholesterolLDL": cholesterol_ldl,
        "CholesterolHDL": cholesterol_hdl,
        "CholesterolTriglycerides": cholesterol_triglycerides,
        "MMSE": mmse,
        "FunctionalAssessment": functional,
        "MemoryComplaints": binary_to_int(memory),
        "BehavioralProblems": binary_to_int(behavior),
        "ADL": adl,
        "Confusion": binary_to_int(confusion),
        "Disorientation": binary_to_int(disorientation),
        "PersonalityChanges": binary_to_int(personality_changes),
        "DifficultyCompletingTasks": binary_to_int(difficulty_tasks),
        "Forgetfulness": binary_to_int(forgetfulness)
    }])
    
    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing patient data..."):
            pred, prob = predict(input_df)
            label_map = {
                0: "Unlikely to have Alzheimer's disease",
                1: "Likely to have Alzheimer's disease"
            }
            
            st.divider()
            st.subheader("🔍 Prediction")
            if pred.iloc[0] == 0:
                st.success(f"✅ {label_map[pred.iloc[0]]}")
                st.metric("Confidence", f"{prob.iloc[0]:.2%}")
            else:
                st.error(f"⚠️ {label_map[pred.iloc[0]]}")
                st.metric("Confidence", f"{prob.iloc[0]:.2%}")

else:
    st.header("📂 Upload Patient Records")
    
    uploaded_file = st.file_uploader("Upload Records of Patients in CSV format", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} patient records")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Processing batch predictions..."):
                preds, probs = predict(df)
                df["PredictionLabel"] = preds
                df["Confidence"] = probs
                
                st.subheader('Results')
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patients", len(df))
                with col2:
                    st.metric("Positive Cases", int(preds.sum()))
                with col3:
                    st.metric("Negative Cases", int(len(preds) - preds.sum()))
                
                st.dataframe(df, use_container_width=True)
                
                csv_buffer = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results",
                    data=csv_buffer,
                    file_name="alzheimers_batch_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )