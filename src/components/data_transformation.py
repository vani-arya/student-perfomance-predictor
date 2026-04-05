"""
data_transformation.py
──────────────────────
Responsible for:
  1. Defining preprocessing pipelines for numerical and categorical features
  2. Fitting the preprocessor on train data, transforming train + test
  3. Saving the preprocessor to artifacts/preprocessor.pkl
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


# ── Config ──────────────────────────────────────────────────
@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ── Component ────────────────────────────────────────────────
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    # ── Build preprocessor ───────────────────────────────────
    def get_preprocessor(self):
        """
        Builds a ColumnTransformer with:
          • Numerical  → Impute (median) → StandardScaler
          • Categorical → Impute (most_frequent) → OneHotEncoder
        """
        try:
            numerical_features = ["reading score", "writing score"]

            categorical_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            # Pipeline for numbers
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ])

            # Pipeline for categories
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("scaler",  StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features),
            ])

            logger.info("Preprocessor built successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ── Run transformation ───────────────────────────────────
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Loads train/test CSVs, applies preprocessing, saves preprocessor.
        Returns: (train_array, test_array, preprocessor_path)
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info(f"Loaded train ({train_df.shape}) and test ({test_df.shape})")

            TARGET = "math score"

            # Separate features and target
            X_train = train_df.drop(columns=[TARGET])
            y_train = train_df[TARGET]
            X_test  = test_df.drop(columns=[TARGET])
            y_test  = test_df[TARGET]

            # Fit on train, transform both
            preprocessor = self.get_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed  = preprocessor.transform(X_test)
            logger.info("Preprocessing applied to train and test sets")

            # Stack features + target into single arrays
            train_array = np.c_[X_train_transformed, np.array(y_train)]
            test_array  = np.c_[X_test_transformed,  np.array(y_test)]

            # Save preprocessor
            save_object(self.config.preprocessor_path, preprocessor)
            logger.info(f"Preprocessor saved → {self.config.preprocessor_path}")

            return train_array, test_array, self.config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
