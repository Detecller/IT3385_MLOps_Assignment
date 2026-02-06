import streamlit as st
import pandas as pd
from mlops_assignment.predict_alzheimer import predict

st.title("🧠 Alzheimer's Disease Prediction (Single)")

st.header("Cognitive & Functional Inputs")

adl = st.slider("ADL (Activities of Daily Living)", 0, 10, 5)
functional = st.slider("Functional Assessment", 0, 10, 5)
memory = st.selectbox("Memory Complaints", [0,1])
behavior = st.selectbox("Behavioral Problems", [0,1])
mmse = st.slider("MMSE", 0, 30, 15)

input_df = pd.DataFrame([{
    "ADL": adl,
    "FunctionalAssessment": functional,
    "MemoryComplaints": memory,
    "BehavioralProblems": behavior,
    "MMSE": mmse
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