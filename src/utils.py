import os
import pickle
import sys

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj) -> None:
    """
    Saves any Python object to a .pkl file using pickle.
    Used to save trained models and preprocessors.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Loads a pickle object from disk.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """
    Trains each model and returns a dict of evaluation metrics.

    Returns:
        {
          "ModelName": {
            "R2": ..., "MAE": ..., "RMSE": ...
          },
          ...
        }
    """
    try:
        results = {}

        for name, model in models.items():
            logger.info(f"Training: {name}")
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test  = r2_score(y_test, y_pred_test)
            mae      = mean_absolute_error(y_test, y_pred_test)
            rmse     = np.sqrt(mean_squared_error(y_test, y_pred_test))

            results[name] = {
                "R2_train": round(r2_train, 4),
                "R2_test":  round(r2_test,  4),
                "MAE":      round(mae,  4),
                "RMSE":     round(rmse, 4),
            }

            logger.info(
                f"{name} → R²(train)={r2_train:.4f} | "
                f"R²(test)={r2_test:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}"
            )

        return results

    except Exception as e:
        raise CustomException(e, sys)
