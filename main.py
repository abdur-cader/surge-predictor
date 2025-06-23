from helpers.training import ModelTrainer
from helpers.prophet_setup import ParcelPredictor
import joblib
import pandas as pd

def main():
    # Example usage
    
    # 1. Train models (only needed once)
    # trainer = ModelTrainer()
    # trainer.load_data('data/prophet_train.xlsx')
    # sample_weights = trainer.preprocess_data()
    # models = trainer.train_models(sample_weights)
    # trainer.save_models(models)
    
    # 2. Predict using Prophet and models
    forecaster = ParcelPredictor()
    results = forecaster.process_data(80)  # predict 80 more rows/days
    results.drop(columns='fleet_available', inplace=True)
    
    # Save results to CSV for inspection
    results.to_csv('data/results/predicted_results.csv', index=False)
    
    # Load preprocessor and model
    preprocessor = joblib.load('data/models/preprocessor.pkl')
    model = joblib.load('data/models/xgb_model.pkl')
    
    # Transform and predict
    results_transformed = preprocessor.transform(results)
    pred = model.predict(results_transformed)
    
    # Add predictions to results
    surge_labels = {
        0: 'High surge',
        1: 'Low surge',
        2: 'Mild surge',
        3: 'No Surge'
    }
    
    results_final = results.copy()
    results_final['predicted_surge_level'] = pred
    results_final['predicted_surge_level'] = results_final['predicted_surge_level'].map(surge_labels)
    
    # Save final predictions
    results_final.to_csv('data/results/final_predictions.csv', index=False)
    print("Predictions saved to data/results/final_predictions.csv")

if __name__ == "__main__":
    main()