# src/components/data_transformation.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import sys
from src.logger import logging
from src.exception import CustomException

class DataTransformer:
    def __init__(self,  output_folder='artifacts'):
        self.train_input_csv = 'artifacts/train.csv'  # Replace with the actual path to your train.csv
        self.test_input_csv = 'artifacts/test.csv'  # Replace with the actual path to your test.csv
        self.output_folder = output_folder

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def label_encode_dataset(self, input_csv, output_csv):
        # Load the dataset
        df = pd.read_csv(input_csv)

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()

        # Iterate through each column and label encode categorical features
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = label_encoder.fit_transform(df[column])

        # Save the transformed dataset
        df.to_csv(output_csv, index=False)
        print(f"Transformed dataset saved to {output_csv}")

    def transform_datasets(self):
        logging.info("Data transformation started.")
        try:
            transformed_train_output_csv = os.path.join(self.output_folder, 'transformed_train.csv')
            transformed_test_output_csv = os.path.join(self.output_folder, 'transformed_test.csv')

            # Apply label encoding to train.csv
            self.label_encode_dataset(self.train_input_csv, transformed_train_output_csv)

            # Apply label encoding to test.csv
            self.label_encode_dataset(self.test_input_csv, transformed_test_output_csv)

            logging.info("Data transformation completed.")
            # Transformed train and test output file paths
        except Exception as e:
            raise CustomException(e, sys)
        
    
