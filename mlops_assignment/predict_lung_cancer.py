from pathlib import Path

import pandas as pd
from pycaret.classification import load_model, predict_model


# Base directory of the repository (one level up from this package)
BASE_DIR = Path(__file__).resolve().parents[1]

# Path to the PyCaret model (without .pkl extension, as expected by load_model)
# This corresponds to `models/lung_cancer_pipeline.pkl` in the repo root.
MODEL_NAME = BASE_DIR / "models" / "lung_cancer_pipeline"

# Load the saved model once at import time
model = load_model(str(MODEL_NAME))


def predict_lung_cancer(input_df: pd.DataFrame) -> dict:
    """
    Run a lung cancer prediction using the PyCaret pipeline.

    Parameters
    ----------
    input_df : pd.DataFrame
        A DataFrame containing one or more rows with the same feature
        columns used during training (all features except the target `Level`).

    Returns
    -------
    dict
        For a single-row input, returns:
        {
            "prediction": <predicted label or value>,
            "probability": <probability score if available, else None>
        }
    """
    prediction_df = predict_model(model, data=input_df)

    # Default keys used by PyCaret for classification workflows
    label_col = "prediction_label"
    score_col = "prediction_score"

    prediction = None
    probability = None

    if label_col in prediction_df.columns:
        prediction = prediction_df[label_col].iloc[0]
    elif "Label" in prediction_df.columns:
        prediction = prediction_df["Label"].iloc[0]

    if score_col in prediction_df.columns:
        probability = float(prediction_df[score_col].iloc[0])

    return {
        "prediction": prediction,
        "probability": probability,
    }

