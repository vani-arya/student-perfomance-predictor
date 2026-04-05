"""
predict_pipeline.py
───────────────────
Loads saved model + preprocessor and predicts for new input data.
Used by the Streamlit app.
"""

import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object


class PredictPipeline:
    """
    Loads the saved preprocessor and model, then predicts on new data.
    """

    MODEL_PATH       = os.path.join("artifacts", "model.pkl")
    PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame) -> float:
        """
        Args:
            features: DataFrame with exactly these columns:
                      gender, race/ethnicity, parental level of education,
                      lunch, test preparation course, reading score, writing score
        Returns:
            Predicted math score (float)
        """
        try:
            logger.info("Loading model and preprocessor for prediction...")
            model        = load_object(self.MODEL_PATH)
            preprocessor = load_object(self.PREPROCESSOR_PATH)

            data_scaled  = preprocessor.transform(features)
            prediction   = model.predict(data_scaled)

            logger.info(f"Prediction: {prediction[0]:.2f}")
            return float(prediction[0])

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    A simple data class to collect user inputs and convert to DataFrame.
    Makes it easy to pass data from the Streamlit UI to the pipeline.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_education: str,
        lunch: str,
        test_prep_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender             = gender
        self.race_ethnicity     = race_ethnicity
        self.parental_education = parental_education
        self.lunch              = lunch
        self.test_prep_course   = test_prep_course
        self.reading_score      = reading_score
        self.writing_score      = writing_score

    def to_dataframe(self) -> pd.DataFrame:
        """Converts inputs to a single-row DataFrame matching training columns."""
        data = {
            "gender":                      [self.gender],
            "race/ethnicity":              [self.race_ethnicity],
            "parental level of education": [self.parental_education],
            "lunch":                       [self.lunch],
            "test preparation course":     [self.test_prep_course],
            "reading score":               [self.reading_score],
            "writing score":               [self.writing_score],
        }
        return pd.DataFrame(data)
