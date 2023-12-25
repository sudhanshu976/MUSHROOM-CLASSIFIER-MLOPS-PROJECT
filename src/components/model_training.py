# src/components/model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from src.logger import logging
from src.exception import CustomException
import sys
import os
import joblib
not_needed = ['class','cap-shape', 'cap-surface', 'cap-color','gill-attachment','stalk-shape', 'stalk-root','stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number']
class ModelTrainer:
    def __init__(self, target_column='class'):
        self.input_csv = 'artifacts/transformed_train.csv'
        self.target_column = target_column
        self.output_folder = "artifacts"

    def load_data(self):
        df = pd.read_csv(self.input_csv)
        X = df.drop(not_needed, axis=1)
        y = df[self.target_column]
        return X, y

    def train_and_evaluate(self):
        logging.info(" Model Training started.")

        try:
        # Load the data
            X, y = self.load_data()

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize XGBoost classifier (you can add hyperparameters as needed)
            model = XGBClassifier()

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            model_filename = os.path.join(self.output_folder, 'trained_model.joblib')
            joblib.dump(model, model_filename)
            logging.info(f"Trained model saved to {model_filename}")

            # Log the evaluation metrics
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")

            logging.info("Model training completed.")   

        

        except Exception as e:
            raise CustomException(e, sys)
    
