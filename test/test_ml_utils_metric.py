
# Test valid inputs and expected outputs
import pytest

from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score


@pytest.mark.parametrize(
    "y_true, y_pred, expected_f1, expected_precision, expected_recall",
    [
        # Perfect predictions
        ([1, 1, 0, 0], [1, 1, 0, 0], 1.0, 1.0, 1.0),
        # All incorrect predictions
        ([1, 1, 0, 0], [0, 0, 1, 1], 0.0, 0.0, 0.0),
        # Partial correct predictions
        ([1, 0, 1, 0], [1, 1, 0, 0], 0.5, 0.5, 0.5),
        # All true negatives
        ([0, 0, 0], [0, 0, 0], 0.0, 0.0, 0.0),
        # Mixed predictions with precision=1.0, recall=0.5, f1≈0.666...
        ([1, 0, 1], [1, 0, 0], pytest.approx(2/3, 0.01), 1.0, 0.5),
    ],
)
def test_get_classification_score_valid_inputs(y_true, y_pred, expected_f1, expected_precision, expected_recall):
    # Act
    result = get_classification_score(y_true, y_pred)

    # Assert
    assert isinstance(result, ClassificationMetricArtifact)
    assert result.f1_score == expected_f1
    assert result.precision_score == expected_precision
    assert result.recall_score == expected_recall

# Test exception handling for invalid inputs
def test_get_classification_score_mismatched_lengths():
    y_true = [1, 0, 1]
    y_pred = [1, 0]  # Mismatched lengths
    with pytest.raises(NetworkSecurityException):
        get_classification_score(y_true, y_pred)

def test_get_classification_score_empty_inputs():
    y_true = []
    y_pred = []  # Empty inputs
    with pytest.raises(NetworkSecurityException):
        get_classification_score(y_true, y_pred)

# Test edge case: all predictions are negative
def test_get_classification_score_all_negatives():
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    result = get_classification_score(y_true, y_pred)
    assert result.f1_score == 0.0
    assert result.precision_score == 0.0
    assert result.recall_score == 0.0

# Test edge case: all predictions are positive
def test_get_classification_score_all_positives():
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]
    result = get_classification_score(y_true, y_pred)
    assert result.f1_score == 1.0
    assert result.precision_score == 1.0
    assert result.recall_score == 1.0