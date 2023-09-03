import pandas as pd
from schema.data_schema import BinaryClassificationSchema
from preprocessing.preprocess import impute_numeric, impute_categorical, encode, normalize
from joblib import load, dump
from config import paths


def run_pipeline(input_data: pd.DataFrame, schema: BinaryClassificationSchema, training: bool = True) -> pd.DataFrame:
    """
        Apply transformations to the input data (Imputations, encoding and normalization).

        Args:
            input_data (pd.DataFrame): Data to be processed.
            schema (BinaryClassificationSchema): BinaryClassificationSchema object carrying data about the schema
            training (bool): Should be set to true if the data is for the training process.
        Returns:
            The data after applying the transformations
        """
    if training:
        imputation_dict = {}
        for f in schema.categorical_features:
            input_data, value = impute_categorical(input_data, f)
            imputation_dict[f] = value

        for f in schema.numeric_features:
            input_data, value = impute_numeric(input_data, f)
            imputation_dict[f] = value

        input_data = normalize(input_data, schema)

        input_data = encode(input_data, schema)
        dump(imputation_dict, paths.IMPUTATION_FILE_PATH)
    else:
        imputation_dict = load(paths.IMPUTATION_FILE_PATH)
        for f in schema.features:
            input_data[f].fillna(imputation_dict.get(f, input_data[f].mode()[0]), inplace=True)
        input_data = normalize(input_data, schema, scaler='predict')
        input_data = encode(input_data, schema, encoder='predict')

    return input_data
