import os
import pandas as pd
import joblib
import numpy as np
from typing import Union
from sklearn.exceptions import NotFittedError
from schema.data_schema import BinaryClassificationSchema
from tpot import TPOTClassifier

PREDICTOR_FILE_NAME = 'predictor.joblib'


def get_config_dict():
    classifier_config_dict = {

        # Classifiers
        'sklearn.tree.DecisionTreeClassifier': {
            'criterion': ["gini", "entropy"],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21)
        },

        'sklearn.ensemble.ExtraTreesClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },

        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },

        'sklearn.ensemble.GradientBoostingClassifier': {
            'n_estimators': [100],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'subsample': np.arange(0.05, 1.01, 0.05),
            'max_features': np.arange(0.05, 1.01, 0.05)
        },

        'sklearn.neighbors.KNeighborsClassifier': {
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
        },

        'sklearn.linear_model.LogisticRegression': {
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False]
        },

        'xgboost.XGBClassifier': {
            'n_estimators': [100],
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': range(1, 21),
            'n_jobs': [1],
            'verbosity': [0]
        },

        'sklearn.neural_network.MLPClassifier': {
            'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
            'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
        },

        # Preprocesssors
        'sklearn.preprocessing.Binarizer': {
            'threshold': np.arange(0.0, 1.01, 0.05)
        },

        'sklearn.decomposition.FastICA': {
            'tol': np.arange(0.0, 1.01, 0.05)
        },

        'sklearn.preprocessing.MaxAbsScaler': {
        },

        'sklearn.preprocessing.MinMaxScaler': {
        },

        'sklearn.preprocessing.Normalizer': {
            'norm': ['l1', 'l2', 'max']
        },


        'sklearn.decomposition.PCA': {
            'svd_solver': ['randomized'],
            'iterated_power': range(1, 11)
        },

        'sklearn.preprocessing.PolynomialFeatures': {
            'degree': [2],
            'include_bias': [False],
            'interaction_only': [False]
        },


        'sklearn.preprocessing.RobustScaler': {
        },

        'sklearn.preprocessing.StandardScaler': {
        },

        'tpot.builtins.ZeroCount': {
        },

        'tpot.builtins.OneHotEncoder': {
            'minimum_fraction': [0.05, 0.1, 0.15, 0.2, 0.25],
            'sparse': [False],
            'threshold': [10]
        },

        # Selectors
        'sklearn.feature_selection.SelectFwe': {
            'alpha': np.arange(0, 0.05, 0.001),
            'score_func': {
                'sklearn.feature_selection.f_classif': None
            }
        },

        'sklearn.feature_selection.SelectPercentile': {
            'percentile': range(1, 100),
            'score_func': {
                'sklearn.feature_selection.f_classif': None
            }
        },

        'sklearn.feature_selection.VarianceThreshold': {
            'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        },

        'sklearn.feature_selection.RFE': {
            'step': np.arange(0.05, 1.01, 0.05),
            'estimator': {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': np.arange(0.05, 1.01, 0.05)
                }
            }
        },

        'sklearn.feature_selection.SelectFromModel': {
            'threshold': np.arange(0, 1.01, 0.05),
            'estimator': {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': np.arange(0.05, 1.01, 0.05)
                }
            }
        }

    }
    return classifier_config_dict


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
                                   config_dict=get_config_dict()
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
