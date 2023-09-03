import numpy as np
import pandas as pd
import warnings
from typing import List
from config import paths
from logger import get_logger
from Classifier import Classifier
from preprocessing.pipeline import run_pipeline
from schema.data_schema import load_json_data_schema
from utils import read_csv_in_directory, save_dataframe_as_csv

warnings.filterwarnings('ignore')

logger = get_logger(task_name="predict")


def create_predictions_dataframe(
        predictions_arr: np.ndarray,
        class_names: List[str],
        prediction_field_name: str,
        ids: pd.Series,
        id_field_name: str,
        return_probs: bool = False,
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Performs the following transformations:
    - converts to pandas dataframe
    - adds class labels as headers for columns containing predicted probabilities
    - inserts the id column

    Args:
        predictions_arr (np.ndarray): Predicted probabilities from predictor model.
        class_names (List[str]): List of target classes (labels).
        prediction_field_name (str): Field name to use for predicted class.
        ids: ids as a numpy array for each of the samples in  predictions.
        id_field_name (str): Name to use for the id field.
        return_probs (bool, optional): If True, returns the predicted probabilities
            for each class. If False, returns the final predicted class for each
            data point. Defaults to False.

    Returns:
        Predictions as a pandas dataframe
    """
    if predictions_arr.shape[1] != len(class_names):
        raise ValueError(
            "Length of class names does not match number of prediction columns"
        )
    predictions_df = pd.DataFrame(predictions_arr, columns=class_names)
    if len(predictions_arr) != len(ids):
        raise ValueError("Length of ids does not match number of predictions")
    predictions_df.insert(0, id_field_name, ids)
    if return_probs:
        return predictions_df
    predictions_df[prediction_field_name] = predictions_df[class_names].idxmax(axis=1)
    predictions_df.drop(class_names, axis=1, inplace=True)
    return predictions_df


def run_batch_predictions(
        test_dir: str = paths.TEST_DIR,
        predictor_dir: str = paths.PREDICTOR_DIR_PATH,
        predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
        input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
        return_proba=False
) -> None:
    """
        Run batch predictions on test data, save the predicted probabilities to a CSV file.

        This function reads test data from the specified directory,
        loads the preprocessing pipeline and pre-trained predictor model,
        transforms the test data using the pipeline,
        makes predictions using the trained predictor model,
        adds ids into the predictions dataframe,
        and saves the predictions as a CSV file.

        Args:
            test_dir (str): Path to the directory containing test data
            predictor_dir (str): Path to the directory containing the predictor model.
            predictions_file_path (str): Path in which to store the prediction file.
            input_schema_dir (str): Path to the schema file of the data
            return_proba (bool): If true, outputs the probabilities of the target classes.
        """
    x_test = read_csv_in_directory(test_dir)

    model = Classifier.load(predictor_dir)
    data_schema = load_json_data_schema(input_schema_dir)
    ids = x_test[data_schema.id]
    x_test = x_test.drop(columns=data_schema.id)
    x_test = run_pipeline(x_test, data_schema, training=False)

    logger.info("Making predictions...")
    predictions_arr = Classifier.predict_with_model(model, x_test, return_proba=True)
    predictions_df = create_predictions_dataframe(
        predictions_arr=predictions_arr,
        class_names=data_schema.target_classes,
        prediction_field_name=data_schema.target,
        ids=ids,
        id_field_name=data_schema.id,
        return_probs=return_proba)
    logger.info("Saving predictions...")
    save_dataframe_as_csv(
        dataframe=predictions_df, file_path=predictions_file_path
    )

    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions(return_proba=True)
