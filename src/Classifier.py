import os
import pandas as pd
import joblib
import numpy as np
from typing import Union
from sklearn.exceptions import NotFittedError
from schema.data_schema import BinaryClassificationSchema
from tpot import TPOTClassifier


PREDICTOR_FILE_NAME = 'predictor.joblib'


class Classifier:
    """A wrapper class for the binary classifier.

        This class provides a consistent interface that can be used with other
        classifier models.
    """

    model_name = 'tpot_binary_classifier'

    def __init__(self, train_input: pd.DataFrame, schema: BinaryClassificationSchema):
        """Construct a new Binary Classifier."""
        self.tpot = TPOTClassifier(generations=5,  # Number of generations for optimization
                                   population_size=20,  # Number of individuals in each generation
                                   verbosity=2,  # Verbosity level (0 to 3)
                                   random_state=42,  # Random seed for reproducibility
                                   n_jobs=-1,  # Number of CPU cores to use (-1 to use all available cores)
                                   )
        self._is_trained: bool = False
        self.train_input = train_input
        self.schema = schema

    def __str__(self):
        return f"Model name: {self.model_name}"

    def train(self) -> None:
        """Train the model on the provided data"""
        x_train = self.train_input.drop(columns=[self.schema.target])
        y_train = self.train_input[self.schema.target]
        self.tpot.fit(x_train, y_train)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            Union[pd.DataFrame, pd.Series]: The output predictions.
        """
        return self.tpot.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.tpot.predict_proba(inputs)

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self.tpot.fitted_pipeline_, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded KNN binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    @classmethod
    def predict_with_model(cls, classifier: "Classifier", data: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Predict class probabilities for the given data.

        Args:
            classifier (Classifier): The classifier model.
            data (pd.DataFrame): The input data.
            return_proba (bool): If true, returns the probabilities of the classes.

        Returns:
            Union[pd.DataFrame, pd.Series]: The output predictions.
        """

        return classifier.predict_proba(data) if return_proba else classifier.predict(data)

    @classmethod
    def save_predictor_model(cls, model: "Classifier", predictor_dir_path: str) -> None:

        """
        Save the classifier model to disk.

        Args:
            model (Classifier): The classifier model to save.
            predictor_dir_path (str): Dir path to which to save the model.
        """
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        model.save(predictor_dir_path)

    @classmethod
    def load_predictor_model(cls, predictor_dir_path: str) -> "Classifier":
        """
        Load the classifier model from disk.

        Args:
            predictor_dir_path (str): Dir path where model is saved.

        Returns:
            Classifier: A new instance of the loaded classifier model.
        """
        return Classifier.load(predictor_dir_path)
