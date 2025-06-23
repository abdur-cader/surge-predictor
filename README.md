# Surge Predictor
---
## Overview
This project predicts parcel delivery surge levels (No Surge, Mild Surge, Low Surge, High Surge) based on historical data using machine learning models and Prophet for time series forecasting.

## Layout
```
.
│   LICENSE
│   main.py
│   README.md
│   requirements.txt
│
├───data
│   │   Prophet_train.xlsx
│   │   results
│   │    └ <results.xlsx>    <--- Results will be stored here by default  
│   └───models
│           cb_model.pkl
│           lgb_model.pkl
│           preprocessor.pkl
│           rf_model.pkl
│           xgb_model.pkl
│           x_test.pkl
│           y_test.pkl
│
└───helpers
        prophet_setup.py
        training.py
```

## Features
The model uses the following features:
- year, month, day
- parcel_count
- day_of_week
- is_weekend
- is_holiday
- is_holiday_soon
- fleet_available
- total_parcel_weight
- avg_parcel_weight
- avg_parcel_volume_size

## Models Trained
1. Random Forest
2. XGBoost
3. LightGBM
4. CatBoost

## Getting Started

### Installation
Install all dependencies:
```bash
pip install -r requirements.txt
```

# Usage
- If you would like to use pre-trained models, you may use joblib to import them from `./data/models/[model].pkl`
1. To train models (run once):
```python
trainer = ModelTrainer()
trainer.load_data('data/prophet_train.xlsx')
sample_weights = trainer.preprocess_data()
models = trainer.train_models(sample_weights)
trainer.save_models(models)
```
2. To make predictions:
```python
forecaster = ParcelPredictor()
results = forecaster.process_data(10)  # predict the next 10 surge levels for each day
results_transformed = preprocessor.transform(results)
predictions = model.predict(results_transformed)
```

