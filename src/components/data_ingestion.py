"""
data_ingestion.py
─────────────────
Responsible for:
  1. Loading the raw dataset (CSV)
  2. Splitting into train / test sets
  3. Saving both splits to artifacts/
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


# ── Config ──────────────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    """Paths for raw, train, and test data artifacts."""
    raw_data_path:   str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path:  str = os.path.join("artifacts", "test.csv")


# ── Component ────────────────────────────────────────────────
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Loads data, splits into train/test, saves to artifacts/.
        Returns: (train_path, test_path)
        """
        logger.info("Starting data ingestion...")
        try:
            # ── Load dataset ──────────────────────────────────
            # Option A: load from local CSV in data/
            data_path = os.path.join("data", "students.csv")

            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                logger.info(f"Loaded dataset from {data_path} — shape: {df.shape}")
            else:
                # Option B: generate synthetic data for demo
                logger.warning(
                    "data/students.csv not found — generating synthetic dataset"
                )
                import numpy as np

                np.random.seed(42)
                n = 1000

                edu_levels = [
                    "high school", "some college", "associate's degree",
                    "bachelor's degree", "master's degree"
                ]
                groups = ["group A", "group B", "group C", "group D", "group E"]

                reading  = np.clip(np.random.normal(69, 15, n).astype(int), 0, 100)
                writing  = np.clip(np.random.normal(68, 15, n).astype(int), 0, 100)
                # Math score correlates with reading/writing + some noise
                math     = np.clip(
                    ((reading + writing) / 2 + np.random.normal(0, 8, n)).astype(int),
                    0, 100
                )

                df = pd.DataFrame({
                    "gender":                      np.random.choice(["male", "female"], n),
                    "race/ethnicity":              np.random.choice(groups, n),
                    "parental level of education": np.random.choice(edu_levels, n),
                    "lunch":                       np.random.choice(["standard", "free/reduced"], n),
                    "test preparation course":     np.random.choice(["none", "completed"], n),
                    "reading score":               reading,
                    "writing score":               writing,
                    "math score":                  math,
                })
                logger.info(f"Synthetic dataset generated — shape: {df.shape}")

            # ── Save raw ──────────────────────────────────────
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved → {self.config.raw_data_path}")

            # ── Train / test split ────────────────────────────
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info(
                f"Train ({len(train_df)}) → {self.config.train_data_path} | "
                f"Test ({len(test_df)})  → {self.config.test_data_path}"
            )

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
