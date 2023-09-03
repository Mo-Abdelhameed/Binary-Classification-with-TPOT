import pandas as pd
import pytest
from src.data_models.prediction_data_model import validate_predictions


@pytest.fixture
def test_key():
    # define the test_key as a pandas DataFrame
    test_key_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "target_field": ["A", "B", "A", "B"],
        }
    )
    return test_key_df


@pytest.fixture
def predictions():
    # define a valid predictions DataFrame
    valid_predictions_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "0": [0.8, 0.1, 0.8, 0.1],
            "1": [0.2, 0.8, 0.1, 0.9]
        }
    )
    return valid_predictions_df


# define a valid predictions DataFrame
valid_predictions = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "0": [0.9, 0.2, 0.8, 0.1, 0.85],
        "1": [0.1, 0.8, 0.2, 0.9, 0.15],
    }
)


def test_validate_data_with_valid_data(schema_provider, predictions):
    """Test the function 'validate_predictions' with valid data."""
    validated_data = validate_predictions(predictions, schema_provider)
    assert validated_data is not None
    assert validated_data.shape == predictions.shape


def test_validate_data_with_missing_id(schema_provider, predictions):
    predictions_missing_id = predictions.drop(columns=["id"])
    with pytest.raises(ValueError) as exc_info:
        _ = validate_predictions(predictions_missing_id, schema_provider)
    assert (
        "ValueError: Malformed predictions file. "
        "ID field 'id' is not present in predictions file" in str(exc_info.value)
    )


def test_validate_data_with_empty_file(schema_provider, predictions):
    empty_predictions = pd.DataFrame(columns=predictions.columns)
    with pytest.raises(ValueError) as exc_info:
        _ = validate_predictions(empty_predictions, schema_provider)
    assert "ValueError: The provided predictions file is empty." in str(exc_info.value)


def test_validate_data_with_missing_class_columns(schema_provider, predictions):
    predictions_missing_class = predictions.drop(columns=["0"])
    with pytest.raises(ValueError) as exc_info:
        _ = validate_predictions(predictions_missing_class, schema_provider)
    assert "ValueError: Malformed predictions file. Target field(s) " in str(
        exc_info.value
    )
    assert "missing in predictions file." in str(exc_info.value)


def test_validate_data_with_invalid_probabilities(schema_provider, predictions):
    predictions_invalid_probabilities = predictions.copy()
    # include invalid probabilities
    predictions_invalid_probabilities["0"] = [-0.1, 0.5, 0.6, 1.2]
    with pytest.raises(ValueError) as exc_info:
        _ = validate_predictions(predictions_invalid_probabilities, schema_provider)
    assert "ValueError: Invalid probabilities in predictions file. " in str(
        exc_info.value
    )
    assert "All probabilities should be between 0 and 1, inclusive." in str(
        exc_info.value
    )