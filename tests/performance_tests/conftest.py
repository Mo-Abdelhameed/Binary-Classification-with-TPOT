import os
import json
import pytest
from src.serve_utils import get_model_resources
from src.train import run_training
from fastapi.testclient import TestClient
from src.serve import create_app


@pytest.fixture
def train_dir(sample_train_data, tmpdir, train_data_file_name):
    """Fixture to create and save a sample DataFrame for testing"""
    train_data_dir = tmpdir.mkdir("train")
    train_data_file_path = train_data_dir.join(train_data_file_name)
    sample_train_data.to_csv(train_data_file_path, index=False)
    return str(train_data_dir)


@pytest.fixture
def input_schema_file_name():
    return "schema.json"


@pytest.fixture
def train_data_file_name():
    return "train.csv"


@pytest.fixture
def input_schema_dir(schema_dict, tmpdir, input_schema_file_name):
    """Fixture to create and save a sample schema for testing"""
    schema_dir = tmpdir.mkdir("input_schema")
    schema_file_path = schema_dir.join(input_schema_file_name)
    with open(schema_file_path, "w") as file:
        json.dump(schema_dict, file)
    return str(schema_dir)


@pytest.fixture
def performance_test_results_dir_path():
    tests_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir_path = os.path.join(tests_dir_path, "test_results", "performance_tests")
    return results_dir_path


@pytest.fixture
def train_predict_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "train_predict_performance_results.csv"
    )
    return str(file_path)


@pytest.fixture
def inference_apis_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "inference_api_performance_results.csv"
    )
    return str(file_path)


@pytest.fixture
def docker_img_build_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "docker_img_build_performance_results.csv"
    )
    return str(file_path)


@pytest.fixture
def resources_paths_dict(test_resources_dir_path):
    """Define a fixture for the paths to the test model resources."""
    return {
        "saved_schema_dir_path": os.path.join(test_resources_dir_path, "schema"),
        "predictor_dir_path": os.path.join(test_resources_dir_path, "predictor"),
    }


@pytest.fixture
def test_resources_dir_path(tmpdir):
    """Define a fixture for the path to the test_resources directory."""
    tmpdir.mkdir("test_resources")
    test_resources_path = os.path.join(tmpdir, "test_resources")
    return test_resources_path


@pytest.fixture
def app(
    input_schema_dir,
    train_dir,
    resources_paths_dict: dict,
):
    """
    Define a fixture for the test app.

    Args:
        input_schema_dir (str): Directory path to the input data schema.
        train_dir (str): Directory path to the training data.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.
    """

    # Create temporary paths for all outputs/artifacts
    saved_schema_dir_path = resources_paths_dict["saved_schema_dir_path"]
    predictor_dir_path = resources_paths_dict["predictor_dir_path"]

    # Run the training process
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_dir_path=saved_schema_dir_path,
        train_dir=train_dir,
        predictor_dir_path=predictor_dir_path,
    )

    # create model resources dictionary
    model_resources = get_model_resources(**resources_paths_dict)

    # create test app
    return TestClient(create_app(model_resources))
