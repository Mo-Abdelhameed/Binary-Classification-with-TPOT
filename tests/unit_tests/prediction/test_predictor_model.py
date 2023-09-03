import os
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from src.Classifier import Classifier
from src.config.paths import PREDICTIONS_FILE_PATH, PREDICTOR_DIR_PATH
from src.preprocessing.pipeline import run_pipeline


def test_train_predict_model(sample_train_data, sample_test_data, schema_provider):
    """
    Test if the classifier is created, trained and makes predictions.
    """
    target = sample_train_data[schema_provider.target]
    sample_train_data = sample_train_data.drop(columns=schema_provider.target)
    sample_train_data = sample_train_data.drop(columns=schema_provider.id)

    sample_train_data = run_pipeline(sample_train_data, schema_provider, training=True)
    sample_train_data[schema_provider.target] = target
    classifier = Classifier(sample_train_data, schema_provider)
    classifier.train()

    sample_test_data = sample_test_data.drop(columns=schema_provider.id)
    sample_test_data = run_pipeline(sample_test_data, schema_provider, training=False)

    predictions = Classifier.predict_with_model(classifier, sample_test_data, return_proba=False)

    assert predictions.shape == (sample_test_data.shape[0],)

    proba_predictions = Classifier.predict_with_model(classifier, sample_test_data, return_proba=True)
    assert proba_predictions.shape == (sample_test_data.shape[0], 2)

    classifier.save(PREDICTOR_DIR_PATH)
    assert os.path.exists(os.path.join(PREDICTOR_DIR_PATH, 'predictor.joblib'))


def test_save_load_model(tmpdir, sample_train_data, sample_test_data, schema_provider):
    """
    Test if the save and load methods work correctly.
    """
    # Specify the file path
    model_dir_path = tmpdir.mkdir("model")
    target = sample_train_data[schema_provider.target]
    sample_train_data = sample_train_data.drop(columns=[schema_provider.target, schema_provider.id])
    sample_train_data = run_pipeline(sample_train_data, schema_provider, training=True)
    sample_train_data[schema_provider.target] = target
    classifier = Classifier(sample_train_data, schema_provider)
    classifier.train()
    # Save the model

    classifier.save(model_dir_path)

    # Load the model
    loaded_clf = Classifier.load(model_dir_path)

    sample_test_data = sample_test_data.drop(columns=schema_provider.id)
    sample_test_data = run_pipeline(sample_test_data, schema_provider, training=False)
    # Test predictions
    predictions = Classifier.predict_with_model(loaded_clf, sample_test_data, return_proba=False)
    assert np.array_equal(predictions, classifier.predict(sample_test_data))

    proba_predictions = Classifier.predict_with_model(loaded_clf, sample_test_data, return_proba=True)
    assert np.array_equal(proba_predictions, classifier.predict_proba(sample_test_data))


def test_classifier_str_representation(classifier):
    """
    Test the `__str__` method of the `Classifier` class.

    The test asserts that the string representation of a `Classifier` instance is
    correctly formatted and includes the model name.

    Args:
        classifier (Classifier): An instance of the `Classifier` class.

    Raises:
        AssertionError: If the string representation of `classifier` does not
        match the expected format.
    """
    classifier_str = str(classifier)

    assert classifier.model_name in classifier_str


def test_predict_with_model(sample_train_data, schema_provider, sample_test_data):
    """
    Test that the 'predict_with_model' function returns predictions of correct size
    and type.
    """
    target = sample_train_data[schema_provider.target]
    sample_train_data = sample_train_data.drop(columns=[schema_provider.target, schema_provider.id])
    sample_train_data = run_pipeline(sample_train_data, schema_provider, training=True)
    sample_train_data[schema_provider.target] = target
    classifier = Classifier(sample_train_data, schema_provider)
    classifier.train()
    sample_test_data = sample_test_data.drop(columns=schema_provider.id)
    sample_test_data = run_pipeline(sample_test_data, schema_provider, training=False)

    predictions = Classifier.predict_with_model(classifier, sample_test_data, return_proba=True)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == sample_test_data.shape[0]


def test_save_predictor_model(tmpdir, sample_train_data, schema_provider):
    """
    Test that the 'save_predictor_model' function correctly saves a Classifier instance
    to disk.
    """
    model_dir_path = os.path.join(tmpdir, "model")
    target = sample_train_data[schema_provider.target]
    sample_train_data = sample_train_data.drop(columns=[schema_provider.target, schema_provider.id])
    sample_train_data = run_pipeline(sample_train_data, schema_provider, training=True)
    sample_train_data[schema_provider.target] = target
    classifier = Classifier(sample_train_data, schema_provider)
    classifier.train()
    Classifier.save_predictor_model(classifier, model_dir_path)
    assert os.path.exists(model_dir_path)
    assert len(os.listdir(model_dir_path)) >= 1


def test_untrained_save_predictor_model_fails(tmpdir, classifier):
    """
    Test that the 'save_predictor_model' function correctly raises  NotFittedError
    when saving an untrained classifier to disk.
    """
    with pytest.raises(NotFittedError):
        model_dir_path = os.path.join(tmpdir, "model")
        Classifier.save_predictor_model(classifier, model_dir_path)
