import pandas as pd
import streamlit as st

from mlops_assignment.predict_lung_cancer import predict_lung_cancer


st.title("🫁 Lung Cancer Risk Prediction")

# Mode selector similar to Alzheimer's page
mode = st.radio("Select Prediction Mode:", ["Single Patient", "Batch Prediction"], horizontal=True)


# Feature names inferred from the original cleaned dataset `Lung_Patient_Cleaned.csv`
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Air_Pollution",
    "Alcohol_use",
    "Dust_Allergy",
    "Occupational_Hazards",
    "Genetic_Risk",
    "Chronic_Lung_Disease",
    "Balanced_Diet",
    "Obesity",
    "Smoking",
    "Passive_Smoker",
    "Chest_Pain",
    "Coughing_of_Blood",
    "Fatigue",
    "Weight_Loss",
    "Shortness_of_Breath",
    "Wheezing",
    "Swallowing_Difficulty",
    "Clubbing_of_Finger_Nails",
    "Frequent_Cold",
    "Dry_Cough",
    "Snoring",
]


if mode == "Single Patient":
    st.header("Patient Features")

    with st.form("lung_cancer_form_single"):
        col1, col2 = st.columns(2)

        inputs = {}

        for i, col in enumerate(FEATURE_COLUMNS):
            current_col = col1 if i % 2 == 0 else col2

            # Simple heuristic: treat Gender as categorical (1/2), others as numeric inputs
            if col == "Gender":
                inputs[col] = current_col.selectbox(col, options=[1, 2], format_func=lambda x: f"{x}")
            else:
                inputs[col] = current_col.number_input(col, value=0.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            input_df = pd.DataFrame([inputs])
            result = predict_lung_cancer(input_df)

            prediction = result.get("prediction", None)
            probability = result.get("probability", None)

            if prediction is None:
                st.error("Prediction could not be computed. Please double-check your inputs.")
            else:
                st.success(f"Predicted risk level: {prediction}")

                if probability is not None:
                    st.write(f"Model confidence: {probability:.2%}")

        except Exception as e:
            st.error(
                "An error occurred while running the prediction. "
                "Please check that all inputs are valid."
            )
            st.exception(e)
else:
    st.header("Upload Patient Records")

    uploaded_file = st.file_uploader("Upload records in CSV format.", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):
            try:
                # Ensure the input data has the required feature columns
                missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
                if missing_cols:
                    st.error(
                        "The uploaded CSV is missing required columns:\n"
                        + ", ".join(missing_cols)
                    )
                else:
                    preds = []
                    probs = []

                    for _, row in df.iterrows():
                        row_df = pd.DataFrame([row[FEATURE_COLUMNS]])
                        result = predict_lung_cancer(row_df)
                        preds.append(result.get("prediction", None))
                        probs.append(result.get("probability", None))

                    df_results = df.copy()
                    df_results["PredictionLabel"] = preds
                    df_results["Confidence"] = probs

                    st.subheader("Results")
                    st.dataframe(df_results)

                    csv_buffer = df_results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Results",
                        data=csv_buffer,
                        file_name="lung_cancer_batch_predictions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(
                    "An error occurred while running batch prediction. "
                    "Please check that the input file is valid."
                )
                st.exception(e)

