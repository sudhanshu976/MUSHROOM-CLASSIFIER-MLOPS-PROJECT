import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

class DataIngestion:
    def __init__(self,test_size=0.2, random_state=42):
        self.input_file = 'notebook/final_data.csv'
        self.test_size = test_size
        self.random_state = random_state
        self.artifacts_folder = 'artifacts'
        self.train_path = os.path.join(self.artifacts_folder, 'train.csv')
        self.test_path = os.path.join(self.artifacts_folder, 'test.csv')

    def create_artifacts_folder(self):
        if not os.path.exists(self.artifacts_folder):
            os.makedirs(self.artifacts_folder)

    def split_and_save_data(self):
        # Load the data from CSV
        data = pd.read_csv(self.input_file)

        # Split the data into train and test sets
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=self.random_state)

        # Save the train and test sets to the artifacts folder
        train_data.to_csv(self.train_path, index=False)
        test_data.to_csv(self.test_path, index=False)

    def run(self):
        try:
            logging.info("Data ingestion and splitting started.")

            # Create the artifacts folder
            self.create_artifacts_folder()

            # Perform data ingestion, splitting, and saving
            self.split_and_save_data()

            logging.info("Data ingestion and splitting completed. Train and test sets saved in the 'artifacts' folder.")
        except Exception as e:
            raise CustomException(e, sys)



        

