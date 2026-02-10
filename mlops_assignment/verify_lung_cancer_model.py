import pandas as pd

from mlops_assignment.predict_lung_cancer import predict_lung_cancer


def main() -> None:
    """
    Simple verification script:
    - constructs a single-row input with all required lung cancer features
    - calls `predict_lung_cancer` and prints the results
    """

    sample = {
        "Age": 50,
        "Gender": 1,
        "Air_Pollution": 3,
        "Alcohol_use": 2,
        "Dust_Allergy": 4,
        "Occupational_Hazards": 3,
        "Genetic_Risk": 4,
        "Chronic_Lung_Disease": 2,
        "Balanced_Diet": 2,
        "Obesity": 3,
        "Smoking": 4,
        "Passive_Smoker": 2,
        "Chest_Pain": 3,
        "Coughing_of_Blood": 2,
        "Fatigue": 3,
        "Weight_Loss": 2,
        "Shortness_of_Breath": 3,
        "Wheezing": 2,
        "Swallowing_Difficulty": 2,
        "Clubbing_of_Finger_Nails": 1,
        "Frequent_Cold": 2,
        "Dry_Cough": 3,
        "Snoring": 2,
    }

    input_df = pd.DataFrame([sample])
    print("Input DataFrame:")
    print(input_df)

    result = predict_lung_cancer(input_df)

    print("\nPrediction result:")
    print(result)


if __name__ == "__main__":
    main()

