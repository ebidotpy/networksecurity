import pytest

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from networksecurity.utils.main_utils.utils import (read_yaml_file, 
                                                    write_yaml_file, 
                                                    save_numpy_array_data, 
                                                    load_numpy_array_data, 
                                                    save_object, 
                                                    load_object, 
                                                    evaluate_models)
import pytest
import yaml
import sys
import os
from networksecurity.exception.exception import NetworkSecurityException
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pickle

@pytest.fixture(scope="session")
def test_session():
    print("Testing has been started")
    print("*=========================================*")

@pytest.fixture(scope="module")
def test_module():
    print("Testing utils has been started")
    print("+-------------------------------------------+")



# Sample YAML content for testing
SAMPLE_YAML_CONTENT = """
key1: value1
key2: value2
"""

# Test for the happy path
def test_read_yaml_file_success(tmp_path):
    # Create a temporary YAML file
    file_path = tmp_path / "test.yaml"
    file_path.write_text(SAMPLE_YAML_CONTENT)

    # Call the function and check the result
    result = read_yaml_file(str(file_path))
    assert result == {"key1": "value1", "key2": "value2"}

# Test for the error path (file not found)
def test_read_yaml_file_file_not_found():
    with pytest.raises(NetworkSecurityException) as exc_info:
        read_yaml_file("non_existent_file.yaml")
    
    assert "non_existent_file.yaml" in str(exc_info.value)

# Test for the error path (invalid YAML content)
def test_read_yaml_file_invalid_yaml(tmp_path):
    # Create a temporary file with invalid YAML content
    file_path = tmp_path / "invalid.yaml"
    file_path.write_text("invalid: yaml: content")

    with pytest.raises(NetworkSecurityException) as exc_info:
        read_yaml_file(str(file_path))
    
    assert "invalid.yaml" in str(exc_info.value)

# Test for the error path (permission denied)
def test_read_yaml_file_permission_denied(tmp_path):
    # Create a temporary YAML file
    file_path = tmp_path / "test.yaml"
    file_path.write_text(SAMPLE_YAML_CONTENT)

    # Mock the open function to raise a PermissionError
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = PermissionError("Permission denied")
        
        with pytest.raises(NetworkSecurityException) as exc_info:
            read_yaml_file(str(file_path))
        
        assert "Permission denied" in str(exc_info.value)



# Sample content to write to YAML file
SAMPLE_CONTENT = {"key1": "value1", "key2": "value2"}

# Test for the happy path (write new file)
def test_write_yaml_file_success(tmp_path):
    file_path = tmp_path / "test.yaml"

    # Call the function
    write_yaml_file(str(file_path), SAMPLE_CONTENT)

    # Verify the file was created and contains the correct content
    assert os.path.exists(file_path)
    with open(file_path, "r") as file:
        content = yaml.safe_load(file)
        assert content == SAMPLE_CONTENT

# Test for the happy path (replace existing file)
def test_write_yaml_file_replace_existing(tmp_path):
    file_path = tmp_path / "test.yaml"
    file_path.write_text("existing content")  # Create a file to replace

    # Call the function with replace=True
    write_yaml_file(str(file_path), SAMPLE_CONTENT, replace=True)

    # Verify the file was replaced and contains the correct content
    assert os.path.exists(file_path)
    with open(file_path, "r") as file:
        content = yaml.safe_load(file)
        assert content == SAMPLE_CONTENT

# Test for the happy path (create directories if they don't exist)
def test_write_yaml_file_create_directories(tmp_path):
    file_path = tmp_path / "subdir" / "test.yaml"

    # Call the function
    write_yaml_file(str(file_path), SAMPLE_CONTENT)

    # Verify the directory and file were created
    assert os.path.exists(file_path)
    with open(file_path, "r") as file:
        content = yaml.safe_load(file)
        assert content == SAMPLE_CONTENT

# Test for the error path (permission denied)
def test_write_yaml_file_permission_denied(tmp_path):
    file_path = tmp_path / "test.yaml"

    # Mock os.makedirs to raise a PermissionError
    with patch("os.makedirs") as mock_makedirs:
        mock_makedirs.side_effect = PermissionError("Permission denied")

        with pytest.raises(NetworkSecurityException) as exc_info:
            write_yaml_file(str(file_path), SAMPLE_CONTENT)

        assert "Permission denied" in str(exc_info.value)

# Test for the error path (invalid content)
# def test_write_yaml_file_invalid_content(tmp_path):
#     file_path = tmp_path / "test.yaml"
#     invalid_content = {"invalid: yaml: content"}  # Invalid content for YAML serialization

#     with pytest.raises(NetworkSecurityException) as exc_info:
#         write_yaml_file(str(file_path), invalid_content)

#     assert "invalid content" in str(exc_info.value).lower()


# Unit tests for save_nompu_array_data
def test_save_numpy_array_data_success(tmpdir):
    file_path = os.path.join(tmpdir, "test_dir", "test_file.npy")
    array = np.array([1, 2, 3])

    save_numpy_array_data(file_path, array)

    assert os.path.exists(file_path)
    loaded_array = np.load(file_path)
    assert np.array_equal(loaded_array, array)

def test_save_numpy_array_data_directory_creation(tmpdir):
    file_path = os.path.join(tmpdir, "new_dir", "test_file.npy")
    array = np.array([1, 2, 3])

    save_numpy_array_data(file_path, array)

    assert os.path.exists(os.path.dirname(file_path))

def test_save_numpy_array_data_exception_handling():
    file_path = "/invalid/path/test_file.npy"
    array = np.array([1, 2, 3])

    with pytest.raises(NetworkSecurityException):
        save_numpy_array_data(file_path, array)

@patch("builtins.open", mock_open())
@patch("numpy.save")
def test_save_numpy_array_data_file_io(mock_np_save, tmpdir):
    file_path = os.path.join(tmpdir, "test_file.npy")
    array = np.array([1, 2, 3])

    save_numpy_array_data(file_path, array)

    mock_np_save.assert_called_once()

@patch("os.makedirs")
def test_save_numpy_array_data_directory_creation_failure(mock_makedirs, tmpdir):
    file_path = os.path.join(tmpdir, "test_file.npy")
    array = np.array([1, 2, 3])
    mock_makedirs.side_effect = Exception("Failed to create directory")

    with pytest.raises(NetworkSecurityException):
        save_numpy_array_data(file_path, array)


# Unit tests for load_numpy_array_data
def test_load_numpy_array_data_success(tmpdir):
    # Create a temporary file with numpy array data
    file_path = os.path.join(tmpdir, "test_file.npy")
    array = np.array([1, 2, 3])
    np.save(file_path, array)

    # Load the array from the file
    loaded_array = load_numpy_array_data(file_path)

    # Verify the loaded array matches the original array
    assert np.array_equal(loaded_array, array)

def test_load_numpy_array_data_file_not_found():
    file_path = "/invalid/path/test_file.npy"

    # Test that the function raises NetworkSecurityException when the file is not found
    with pytest.raises(NetworkSecurityException):
        load_numpy_array_data(file_path)

@patch("builtins.open", mock_open())
@patch("numpy.load")
def test_load_numpy_array_data_file_io(mock_np_load):
    file_path = "test_file.npy"
    mock_np_load.return_value = np.array([1, 2, 3])

    # Call the function
    loaded_array = load_numpy_array_data(file_path)

    # Verify that np.load was called and the returned array is correct
    mock_np_load.assert_called_once()
    assert np.array_equal(loaded_array, np.array([1, 2, 3]))

@patch("builtins.open", side_effect=Exception("File read error"))
def test_load_numpy_array_data_exception_handling(mock_open):
    file_path = "test_file.npy"

    # Test that the function raises NetworkSecurityException when an exception occurs
    with pytest.raises(NetworkSecurityException):
        load_numpy_array_data(file_path)


# Unit test for save_object

def test_save_object_success(tmpdir):
    file_path = os.path.join(tmpdir, "test_dir", "test_file.pkl")
    obj = {"key": "value"}

    save_object(file_path, obj)

    assert os.path.exists(file_path)
    with open(file_path, "rb") as file_obj:
        loaded_obj = pickle.load(file_obj)
    assert loaded_obj == obj

def test_save_object_directory_creation(tmpdir):
    file_path = os.path.join(tmpdir, "new_dir", "test_file.pkl")
    obj = {"key": "value"}

    save_object(file_path, obj)

    assert os.path.exists(os.path.dirname(file_path))

def test_save_object_exception_handling():
    file_path = "/invalid/path/test_file.pkl"
    obj = {"key": "value"}

    with pytest.raises(NetworkSecurityException):
        save_object(file_path, obj)

@patch("builtins.open", mock_open())
@patch("pickle.dump")
def test_save_object_file_io(mock_pickle_dump, tmpdir):
    file_path = os.path.join(tmpdir, "test_file.pkl")
    obj = {"key": "value"}

    save_object(file_path, obj)

    mock_pickle_dump.assert_called_once()

@patch("os.makedirs")
def test_save_object_directory_creation_failure(mock_makedirs, tmpdir):
    file_path = os.path.join(tmpdir, "test_file.pkl")
    obj = {"key": "value"}
    mock_makedirs.side_effect = Exception("Failed to create directory")

    with pytest.raises(NetworkSecurityException):
        save_object(file_path, obj)

@patch("logging.info")
def test_save_object_logging(mock_logging_info, tmpdir):
    file_path = os.path.join(tmpdir, "test_file.pkl")
    obj = {"key": "value"}

    save_object(file_path, obj)

    mock_logging_info.assert_any_call("Entered the save_object method of MainUtils class")
    mock_logging_info.assert_any_call("Exited the save_object method of MainUtils class")


# Unit tests for load_object
def test_load_object_success(tmpdir):
    # Create a temporary file with a pickled object
    file_path = os.path.join(tmpdir, "test_file.pkl")
    obj = {"key": "value"}
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

    # Load the object from the file
    loaded_obj = load_object(file_path)

    # Verify the loaded object matches the original object
    assert loaded_obj == obj

def test_load_object_file_not_found():
    file_path = "/invalid/path/test_file.pkl"

    # Test that the function raises NetworkSecurityException when the file is not found
    with pytest.raises(NetworkSecurityException):
        load_object(file_path)

@patch("os.path.exists", return_value=True)  # Mock os.path.exists to return True
@patch("builtins.open", mock_open())  # Mock the open function
@patch("pickle.load")  # Mock pickle.load
def test_load_object_file_io(mock_pickle_load, mock_exists):
    file_path = "test_file.pkl"
    mock_pickle_load.return_value = {"key": "value"}  # Set the return value for pickle.load

    # Call the function
    loaded_obj = load_object(file_path)

    # Verify that pickle.load was called and the returned object is correct
    mock_pickle_load.assert_called_once()
    assert loaded_obj == {"key": "value"}

@patch("builtins.open", side_effect=Exception("File read error"))
def test_load_object_exception_handling(mock_open):
    file_path = "test_file.pkl"

    # Test that the function raises NetworkSecurityException when an exception occurs
    with pytest.raises(NetworkSecurityException):
        load_object(file_path)

@patch("os.path.exists", return_value=False)
def test_load_object_file_not_exists(mock_exists):
    file_path = "test_file.pkl"

    # Test that the function raises NetworkSecurityException when the file does not exist
    with pytest.raises(NetworkSecurityException):
        load_object(file_path)

@patch("builtins.print")
def test_load_object_print_statement(mock_print, tmpdir):
    # Create a temporary file with a pickled object
    file_path = os.path.join(tmpdir, "test_file.pkl")
    obj = {"key": "value"}
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

    # Load the object from the file
    load_object(file_path)

    # Verify that the print statement was called
    mock_print.assert_called_once()

# Unit test for evaluate_models

def test_evaluate_models_success():
    # Create dummy data
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([1, 2, 3])
    X_test = np.array([[4], [5]])
    y_test = np.array([4, 5])

    # Define models and parameters
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor()
    }
    param = {
        "LinearRegression": {"fit_intercept": [True, False]},
        "DecisionTreeRegressor": {"max_depth": [1, 2]}
    }

    # Call the function
    report = evaluate_models(X_train, y_train, X_test, y_test, models, param)

    # Verify the output report
    assert isinstance(report, dict)
    assert "LinearRegression" in report
    assert "DecisionTreeRegressor" in report
    assert isinstance(report["LinearRegression"], float)
    assert isinstance(report["DecisionTreeRegressor"], float)

@patch("sklearn.model_selection.GridSearchCV.fit", side_effect=Exception("GridSearchCV failed"))
def test_evaluate_models_gridsearchcv_failure(mock_fit):
    # Create dummy data
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([1, 2, 3])
    X_test = np.array([[4], [5]])
    y_test = np.array([4, 5])

    # Define models and parameters
    models = {
        "LinearRegression": LinearRegression(),
    }
    param = {
        "LinearRegression": {"fit_intercept": [True, False]},
    }

    # Test that the function raises NetworkSecurityException when GridSearchCV fails
    with pytest.raises(NetworkSecurityException):
        evaluate_models(X_train, y_train, X_test, y_test, models, param)

# @patch("sklearn.metrics.r2_score", side_effect=Exception("r2_score failed"))
# def test_evaluate_models_r2_score_failure(mock_r2_score):
#     # Create dummy data
#     X_train = np.array([[1], [2], [3]])
#     y_train = np.array([1, 2, 3])
#     X_test = np.array([[4], [5]])
#     y_test = np.array([4, 5])

#     # Define models and parameters
#     models = {
#         "LinearRegression": LinearRegression(),
#     }
#     param = {
#         "LinearRegression": {"fit_intercept": [True, False]},
#     }

#     # Mock the r2_score call after GridSearchCV
#     with patch("sklearn.metrics.r2_score", side_effect=Exception("r2_score failed")):
#         # Test that the function raises NetworkSecurityException when r2_score fails
#         with pytest.raises(NetworkSecurityException):
#             evaluate_models(X_train, y_train, X_test, y_test, models, param)

def test_evaluate_models_empty_models():
    # Create dummy data
    X_train = np.array([[1], [2], [3]])
    y_train = np.array([1, 2, 3])
    X_test = np.array([[4], [5]])
    y_test = np.array([4, 5])

    # Define empty models and parameters
    models = {}
    param = {}

    # Call the function
    report = evaluate_models(X_train, y_train, X_test, y_test, models, param)

    # Verify the output report is empty
    assert isinstance(report, dict)
    assert len(report) == 0



