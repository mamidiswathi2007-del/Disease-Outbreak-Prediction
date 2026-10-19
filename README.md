# Disease Outbreak Prediction Using Machine Learning

## Project Overview
This project demonstrates a Machine Learning-based
approach to classifying disease outbreak data using
temperature, rainfall, and reported cases.

## Objectives
- Analyze disease-related data.
- Train a Machine Learning classification model.
- Predict an outbreak class from user inputs.
- Display results through an interactive dashboard.

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib

## Project Structure
- app.py: Streamlit dashboard
- train_model.py: Model training script
- data/disease_data.csv: Sample dataset
- models/disease_model.pkl: Trained model
- assets/style.css: Dashboard styling

## How to Run

1. Install the required libraries:
   pip install -r requirements.txt

2. Train the model:
   python train_model.py

3. Run the dashboard:
   streamlit run app.py

## Disclaimer
This project uses a small synthetic dataset for
educational demonstration only. The model has not
been validated for real-world disease outbreak
prediction or public-health decisions.