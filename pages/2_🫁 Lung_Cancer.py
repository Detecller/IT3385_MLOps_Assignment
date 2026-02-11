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

EXPOSURE_SCALE = {
    1: "None",
    2: "Very Low",
    3: "Low",
    4: "Mild",
    5: "Moderate",
    6: "High",
    7: "Very High",
    8: "Extreme",
}

SYMPTOM_SEVERITY_SCALE = {
    1: "None",
    2: "Very Mild",
    3: "Mild",
    4: "Slight",
    5: "Moderate",
    6: "Noticeable",
    7: "Severe",
    8: "Very Severe",
    9: "Critical",
}

BALANCED_DIET_SCALE = {
    1: "Very Poor Diet",
    2: "Poor",
    3: "Below Average",
    4: "Average",
    5: "Slightly Healthy",
    6: "Healthy",
    7: "Very Healthy",
    8: "Excellent",
}

GENDER_SCALE = {
    1: "Male",
    2: "Female",
}


def labeled_selectbox(label, mapping_dict, default_value_int):
    labels = list(mapping_dict.values())
    default_label = mapping_dict.get(default_value_int, labels[0])
    selected_label = st.selectbox(label, options=labels, index=labels.index(default_label))
    inverse_map = {v: k for k, v in mapping_dict.items()}
    return inverse_map[selected_label]


EXPOSURE_COLUMNS = {
    "Air_Pollution",
    "Alcohol_use",
    "Dust_Allergy",
    "Occupational_Hazards",
    "Genetic_Risk",
    "Chronic_Lung_Disease",
    "Smoking",
    "Passive_Smoker",
    "Obesity",
}

SYMPTOM_COLUMNS = {
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
}


if mode == "Single Patient":
    st.header("Patient Features")

    with st.form("lung_cancer_form_single"):
        col1, col2 = st.columns(2)

        inputs = {}

        for i, col in enumerate(FEATURE_COLUMNS):
            current_col = col1 if i % 2 == 0 else col2

            with current_col:
                if col == "Age":
                    inputs[col] = st.number_input(col, value=0, min_value=0, step=1)
                elif col == "Gender":
                    inputs[col] = labeled_selectbox(col, GENDER_SCALE, default_value_int=1)
                elif col == "Balanced_Diet":
                    inputs[col] = labeled_selectbox(col, BALANCED_DIET_SCALE, default_value_int=4)
                elif col in EXPOSURE_COLUMNS:
                    inputs[col] = labeled_selectbox(col, EXPOSURE_SCALE, default_value_int=1)
                elif col in SYMPTOM_COLUMNS:
                    inputs[col] = labeled_selectbox(col, SYMPTOM_SEVERITY_SCALE, default_value_int=1)
                else:
                    inputs[col] = st.number_input(col, value=0.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            features_dict = {col: inputs[col] for col in FEATURE_COLUMNS}
            input_df = pd.DataFrame([features_dict])
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

