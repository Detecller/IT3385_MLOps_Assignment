import streamlit as st
import pandas as pd
from mlops_assignment.predict_alzheimer import predict

st.title("🧠 Alzheimer's Disease Prediction (Batch)")

uploaded_file = st.file_uploader("Upload Records of Patients in CSV format.", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):

        preds, probs = predict(df)

        df["PredictionLabel"] = preds
        df["Confidence"] = probs

        st.subheader('Results')
        st.dataframe(df)

        csv_buffer = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Results",
            data=csv_buffer,
            file_name="alzheimers_batch_predictions.csv",
            mime="text/csv"
        )