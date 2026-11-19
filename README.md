# Credit Risk Scoring System

A Machine Learning Based Credit Risk Prediction Application built using Python and Streamlit.

## Live Demo

[Click here to try the Credit Risk Scoring System](https://credit-risk-scoring-3rcisp32qmpcqkglfselfu.streamlit.app/)

## Project Overview

This project predicts whether a loan applicant belongs to the Good Credit or Bad Credit category based on their financial and personal information.

The system uses a trained Random Forest classification model to generate predictions and estimated probabilities.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Dataset

- Dataset: German Credit Data
- Source: UCI Machine Learning Repository
- 1,000 records
- 20 input features

## Machine Learning Workflow

1. Dataset loading
2. Data preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature encoding and scaling
5. Model training
6. Hyperparameter tuning
7. Model evaluation
8. Feature importance analysis
9. Streamlit application development
10. Cloud deployment

## Models Used

- Logistic Regression
- Random Forest Classifier
- Tuned Random Forest Classifier

## Application Features

- Interactive applicant information form
- Good Credit / Bad Credit prediction
- Estimated credit risk probabilities
- Applicant information summary
- Live web application

## Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Files

- `app.py` – Streamlit application
- `requirements.txt` – Required Python libraries
- `tuned_random_forest_model.pkl` – Trained model
- `scaler.pkl` – Feature scaling object
- `encoder.pkl` – Categorical feature encoder
- `project_metadata.pkl` – Project metadata

## Disclaimer

This project is developed for educational and demonstration purposes only. Its predictions are not intended for actual lending or financial decisions.

## Author

**Jaid Ali Khan**

B.Tech CSE – Artificial Intelligence and Machine Learning
