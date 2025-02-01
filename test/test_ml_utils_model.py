import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
 # Replace 'your_module' with the actual module path

# Fixture for creating a preprocessor and model
@pytest.fixture
def setup_preprocessor_and_model():
    preprocessor = StandardScaler()
    model = LogisticRegression()
    # Fit the preprocessor and model with dummy data
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    preprocessor.fit(X_train)
    model.fit(preprocessor.transform(X_train), y_train)
    return preprocessor, model

# Test initialization of NetworkModel
def test_network_model_initialization(setup_preprocessor_and_model):
    preprocessor, model = setup_preprocessor_and_model
    network_model = NetworkModel(preprocessor, model)
    assert network_model.preprocessor == preprocessor
    assert network_model.model == model

# Test prediction with valid input
def test_network_model_predict(setup_preprocessor_and_model):
    preprocessor, model = setup_preprocessor_and_model
    network_model = NetworkModel(preprocessor, model)
    X_test = np.array([[1, 2], [3, 4]])
    predictions = network_model.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)

# Test initialization with invalid preprocessor
def test_network_model_initialization_invalid_preprocessor():
    invalid_preprocessor = None
    model = LogisticRegression()
    with pytest.raises(NetworkSecurityException):
        NetworkModel(invalid_preprocessor, model)

# Test initialization with invalid model
def test_network_model_initialization_invalid_model(setup_preprocessor_and_model):
    preprocessor, _ = setup_preprocessor_and_model
    invalid_model = None
    with pytest.raises(NetworkSecurityException):
        NetworkModel(preprocessor, invalid_model)

# Test prediction with invalid input
def test_network_model_predict_invalid_input(setup_preprocessor_and_model):
    preprocessor, model = setup_preprocessor_and_model
    network_model = NetworkModel(preprocessor, model)
    invalid_X_test = None
    with pytest.raises(NetworkSecurityException):
        network_model.predict(invalid_X_test)

# Test prediction with input that cannot be transformed
def test_network_model_predict_transform_error(setup_preprocessor_and_model):
    preprocessor, model = setup_preprocessor_and_model
    network_model = NetworkModel(preprocessor, model)
    X_test = np.array([[1, 2, 3]])  # Incorrect number of features
    with pytest.raises(NetworkSecurityException):
        network_model.predict(X_test)