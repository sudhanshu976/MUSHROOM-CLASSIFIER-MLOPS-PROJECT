# src/components/model_evaluation.py
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from src.logger import logging
import joblib
import sys
from src.exception import CustomException
not_needed = ['class','cap-shape', 'cap-surface', 'cap-color','gill-attachment','stalk-shape', 'stalk-root','stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number']
class ModelEvaluator:
    def __init__(self, target_column='class'):
        self.model_filename = 'artifacts/trained_model.joblib'
        self.input_csv = 'artifacts/transformed_test.csv'
        self.target_column = target_column

    def load_data(self):
        df = pd.read_csv(self.input_csv)
        X = df.drop(not_needed, axis=1)
        y = df[self.target_column]
        return X, y

    def load_model(self):
        # Load the saved model
        model = joblib.load(self.model_filename)
        return model

    def evaluate_model(self):
        try:
        # Load the transformed test data
            X_test, y_test = self.load_data()

            # Load the saved model
            model = self.load_model()

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Display the accuracy
            logging.info(f"Model Accuracy on Test Data: {accuracy:.4f}")
        except Exception as e:
            raise CustomException(e, sys)

    
