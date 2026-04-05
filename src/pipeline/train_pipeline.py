"""
train_pipeline.py
─────────────────
Orchestrates the full training flow:
  DataIngestion → DataTransformation → ModelTrainer

Run with:
  python -m src.pipeline.train_pipeline
"""

import sys
from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    try:
        logger.info("=" * 60)
        logger.info("      STUDENT PERFORMANCE PREDICTOR — TRAINING PIPELINE")
        logger.info("=" * 60)

        # Step 1 — Data Ingestion
        logger.info("STEP 1: Data Ingestion")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Step 2 — Data Transformation
        logger.info("STEP 2: Data Transformation")
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )

        # Step 3 — Model Training
        logger.info("STEP 3: Model Training")
        trainer = ModelTrainer()
        best_name, best_r2, results = trainer.initiate_model_training(train_arr, test_arr)

        logger.info("=" * 60)
        logger.info(f"✅ Training complete!")
        logger.info(f"   Best Model : {best_name}")
        logger.info(f"   R² (test)  : {best_r2:.4f}")
        logger.info(f"   Model saved: artifacts/model.pkl")
        logger.info(f"   Preprocessor: {preprocessor_path}")
        logger.info("=" * 60)

        return best_name, best_r2, results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
