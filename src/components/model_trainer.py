"""
model_trainer.py
────────────────
Responsible for:
  1. Training multiple regression models
  2. Comparing them using R², MAE, RMSE
  3. Selecting the best model
  4. Saving the best model to artifacts/model.pkl
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import evaluate_models, save_object


# ── Config ──────────────────────────────────────────────────
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    r2_threshold: float = 0.60  # Minimum acceptable R² score


# ── Component ────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        Trains all models, compares, picks the best, saves it.
        Returns: (best_model_name, best_r2, results_dict)
        """
        try:
            logger.info("Splitting arrays into X/y...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test,  y_test  = test_array[:, :-1],  test_array[:, -1]

            # ── Define models ────────────────────────────────
            models = {
                "Linear Regression":       LinearRegression(),
                "Decision Tree":           DecisionTreeRegressor(random_state=42),
                "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=42),
            }

            # ── Evaluate all models ──────────────────────────
            logger.info("Evaluating all models...")
            results = evaluate_models(X_train, y_train, X_test, y_test, models)

            # ── Print comparison table ───────────────────────
            logger.info("\n" + "=" * 65)
            logger.info(f"{'Model':<25} {'R²(train)':>10} {'R²(test)':>10} {'MAE':>8} {'RMSE':>8}")
            logger.info("-" * 65)
            for name, metrics in results.items():
                logger.info(
                    f"{name:<25} {metrics['R2_train']:>10} {metrics['R2_test']:>10} "
                    f"{metrics['MAE']:>8} {metrics['RMSE']:>8}"
                )
            logger.info("=" * 65)

            # ── Select best by test R² ───────────────────────
            best_model_name = max(results, key=lambda k: results[k]["R2_test"])
            best_r2         = results[best_model_name]["R2_test"]

            logger.info(f"Best model: {best_model_name} — R²(test) = {best_r2:.4f}")

            if best_r2 < self.config.r2_threshold:
                raise CustomException(
                    f"No model met the R² threshold of {self.config.r2_threshold}. "
                    f"Best was {best_model_name} at {best_r2:.4f}",
                    sys
                )

            # Re-train best model (evaluate_models already trained it, but re-fit cleanly)
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            # ── Save best model ──────────────────────────────
            save_object(self.config.model_path, best_model)
            logger.info(f"Best model saved → {self.config.model_path}")

            return best_model_name, best_r2, results

        except Exception as e:
            raise CustomException(e, sys)
