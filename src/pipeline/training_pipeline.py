from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluator


if __name__ == "__main__":
    # Create an instance of the DataIngestion class
    data_ingestion = DataIngestion()
    # Run the data ingestion process
    data_ingestion.run()

    # Create an instance of DataTransformer
    data_transformer = DataTransformer()
    # Transform datasets
    data_transformer.transform_datasets()

    # Create an instance of ModelTrainer
    model_trainer = ModelTrainer()
    # Train and evaluate the model
    model_trainer.train_and_evaluate()

    # Create an instance of ModelEvaluator
    model_evaluator = ModelEvaluator()
    # Evaluate the model on the transformed test data
    model_evaluator.evaluate_model()
