
"""Model evaluation functions for MMA3001."""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix
)


def evaluate_model(
    y_true, y_scores, threshold=0.5
):
    """Evaluate binary defect predictions.

    Args:
        y_true: Actual binary labels.
        y_scores: Continuous regression predictions.
        threshold: Classification threshold.

    Returns:
        Dictionary of regression and classification
        performance metrics.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if y_true.shape != y_scores.shape:
        raise ValueError("Labels and scores must match")

    y_pred = (y_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "MAE": mean_absolute_error(
            y_true, y_scores
        ),
        "MSE": mean_squared_error(
            y_true, y_scores
        ),
        "Accuracy": accuracy_score(
            y_true, y_pred
        ),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }

