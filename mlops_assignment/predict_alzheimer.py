import pandas as pd
from pycaret.classification import load_model, predict_model


# Load the saved model
MODEL_PATH = "models/alzheimer_pred_model"
model = load_model(MODEL_PATH)


def predict(input_df: pd.DataFrame) -> pd.DataFrame:

    prediction_df = predict_model(model, data=input_df)

    predicted_class = prediction_df['prediction_label'].iloc[0]
    predicted_prob = prediction_df['prediction_score'].iloc[0]
    
    return predicted_class, predicted_prob