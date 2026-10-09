# House Price Prediction Using Machine Learning

## Project Overview

This project predicts house prices in Indian cities using machine learning.
It uses property features such as city, property type, area, number of
bedrooms, bathrooms, and property age to estimate property prices.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Machine Learning Models

The project compares the following algorithms:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Random Forest is used for the final prediction model.

## Project Structure

- `app.py` - Streamlit web application
- `train_model.py` - Trains and evaluates the model
- `compare_models.py` - Compares machine learning algorithms
- `visualize_data.py` - Generates data visualizations
- `requirements.txt` - Required Python libraries
- `data/` - Dataset files (not included in this repository)
- `models/` - Trained model files
- `visualizations/` - Generated charts

## Dataset

The project uses the Indian Urban House Price Prediction (Synthetic)
dataset from Kaggle.

Dataset source:
https://www.kaggle.com/competitions/indian-urban-house-price-prediction

The dataset is not included in this repository. Please obtain it through
Kaggle and review its competition rules before using or redistributing it.

## Installation

Install the required libraries:


pip install -r requirements.txt