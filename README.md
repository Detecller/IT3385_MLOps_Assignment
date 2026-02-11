# MLOps Assignment

## Team Members

- Winston Pawitra Lystanto
- Chew Rui Hong

## Tasks

- Perform an Exploratory Data Analysis on a Given Dataset
- Train, validate and develop a Machine Learning pipeline using PyCaret.
- Build and deploy a front-end web application with real-time prediction
- Set up development and deployment environment according to MLOps Lifecycle

## Web Application

This app hosts two models to predict whether one has **Alzheimer's Disease** and **Lung Cancer** separately.

Batch uploading was implemented to support the prediction of multiple patient records at once, allowing users to submit a CSV file with many entries, automatically generate predictions and confidence scores for each record, and download the results in a single file.

🔗 Live Link: https://it3385-mlops-assignment.onrender.com/

## 🚀 Deployment Guide

- Create a new project on Render.
- Create a new web service within the project.
- Connect the web service to a repository to enable continuous deployment.

## 📘 User Guide

### Link to Repository

https://github.com/Detecller/IT3385_MLOps_Assignment

### How to run with dependencies

Go to root to run the following commands:

```
Jupyter Notebook: poetry run jupyter notebook
MLFlow: poetry run mlflow ui
Streamlit: poetry run streamlit run Introduction.py
```

To add dependencies, use `poetry add <libraries, separated by space>`.

### Data Folder

- Save cleaned data in `data/processed` - via jupyter notebook code
- Save initial data from kaggle in `data/raw` - upload manually

### Notebook (IPYNB)

Use respective folder in `notebooks`.

### Save Model

Save to the `models` folder (e.g. `exp2.save_model(final_model, '../../models/alzheimer_pred_model'`).

### Creating Predict Function

- The purpose of creating one is to serve as an intermediary for the model to generate the predictions, so the code is more modular.
- This function will be used in the Streamlit page to obtain the predictions.

Create the python file in `mlops_assignment`. Take `predict_alzheimer.py` as reference.

### Creating Streamlit Pages

Create streamlit file in `pages`, adding a number and underscore before each file name. This determines the order of the pages when rendered in the UI.
