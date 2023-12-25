# src/pipeline/predict_pipeline.py

from flask import jsonify,render_template
import joblib  # Use joblib for compatibility with older scikit-learn versions
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys

class PredictPipeline:
    def __init__(self, model_path='artifacts/trained_model.joblib'):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        logging.info("Prediction started.")
        try:
            # Convert input_data to a numpy array if needed
            input_data = np.array(input_data).reshape(1, -1)

            # Make prediction using the model
            prediction = self.model.predict(input_data)

            # Map the prediction to a human-readable label (assuming 0 is one class and 1 is another)
            result_label = "Poisonous" if prediction[0] == 1 else "Edible"

            logging.info(f"Prediction completed. Result: {result_label}")

            return result_label
            # Return the result as a dictionary
        
        except Exception as e:
            raise CustomException(e, sys)
