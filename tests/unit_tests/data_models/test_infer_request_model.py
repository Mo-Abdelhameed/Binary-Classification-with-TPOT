import pytest
from pydantic import ValidationError

from data_models.infer_request_model import (
    create_instance_model,
    get_inference_request_body_model,
)


def test_can_create_instance_model(schema_provider):
    """
    Test creation of the instance model with valid schema provider.

    Ensures that a valid instance model is created without raising an exception.
    """
    try:
        _ = create_instance_model(schema_provider)
    except Exception as e:
        pytest.fail(f"Instance model creation failed with exception: {e}")


@pytest.fixture
def SampleInstanceModel(schema_provider):
    InstanceModel = create_instance_model(schema_provider)
    return InstanceModel


def test_valid_instance(SampleInstanceModel):
    """
    Test the instance model with valid instances.
    """
    # valid instance
    try:
        _ = SampleInstanceModel.parse_obj(
            {
                "id": "1232",
                "numeric_feature_1": 50,
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )
    except Exception as e:
        pytest.fail(f"Instance parsing failed with exception: {e}")

    # valid instance with extra feature (still valid)
    try:
        _ = SampleInstanceModel.parse_obj(
            {
                "id": "1232",
                "numeric_feature_1": 50,
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
                "extra_feature": 0.5,
            }
        )
    except Exception as e:
        pytest.fail(f"Instance parsing failed with exception: {e}")


def test_invalid_instance(SampleInstanceModel):
    """
    Test the instance model with invalid instances.

    Ensures that instance model validation raises an exception.
    """
    # empty instance
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj({})

    # missing feature_1
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "id": "1232",
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )

    # missing all features
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj({"id": "1232"})

    # missing id
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "numeric_feature_1": 50,
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )

    # id is None
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "id": None,
                "numeric_feature_1": 50,
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )

    # missing id and feature
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )

    # wrong data type for numeric_feature_1
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "id": "1232",
                "numeric_feature_1": "invalid",
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )


def test_get_inference_request_body_model(schema_provider):
    """
    Test creation of the instance model with valid schema provider.

    Ensures that a valid instance model is created without raising an exception.
    """
    try:
        _ = get_inference_request_body_model(schema_provider)
    except Exception as e:
        pytest.fail(f"Request Body model creation failed with exception: {e}")


@pytest.fixture
def SampleRequestBodyModel(schema_provider):
    InferenceRequestBody = get_inference_request_body_model(schema_provider)
    return InferenceRequestBody


def test_valid_inference_request_body(SampleRequestBodyModel):
    """
    Test the inference request body model with valid data.

    Ensures that a valid request body model is created without raising an exception.
    """
    # valid request with single instance
    try:
        # valid request body
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "1232",
                        "numeric_feature_1": 50,
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "B",
                    }
                ]
            }
        )
    except Exception as e:
        pytest.fail(f"Inference request body parsing failed with exception: {e}")

    # valid request with multiple instances
    try:
        # valid request body
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "123",
                        "numeric_feature_1": 50,
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "B",
                    },
                    {
                        "id": "456",
                        "numeric_feature_1": 60,
                        "numeric_feature_2": 1.5,
                        "categorical_feature_1": "B",
                        "categorical_feature_2": "C",
                    },
                ]
            }
        )
    except Exception as e:
        pytest.fail(f"Inference request body parsing failed with exception: {e}")

    # valid request - extra key
    try:
        # valid request body
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "1232",
                        "numeric_feature_1": 50,
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "D",
                    }
                ],
                "extra": "key",
            }
        )
    except Exception as e:
        pytest.fail(f"Inference request body parsing failed with exception: {e}")


def test_invalid_inference_request_body(SampleRequestBodyModel):
    """
    Test the inference request body model with invalid instance(s).

    - Test the inference request body model with missing 'instances'.
    - Test the inference request body model with empty 'instances'.
    - Test the inference request body model with empty 'instances'.

    Ensures that request body model validation raises an exception.
    """
    # request is empty
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj({})

    # 'instances' is not a list
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj({"instances": "invalid"})

    # 'instances' is empty
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj({"instances": []})

    # 'instances' has sample instance with missing feature
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "1232",
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "A",
                    }
                ]
            }
        )

    # 'instances' has valid and invalid sample
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "123",
                        "numeric_feature_1": 50,
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "B",
                    },
                    {
                        "id": "456",
                        "numeric_feature_1": 60,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "B",
                    },
                ]
            }
        )
