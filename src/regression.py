
"""Regression models for packaging defect analysis."""

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def train_linear_regression(X_train, y_train):
    """Train a linear regression baseline.

    Args:
        X_train: Numerical training features.
        y_train: Binary defect labels (0 or 1).

    Returns:
        Fitted LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train, y_train, max_depth=5
):
    """Train a decision tree regression model.

    Args:
        X_train: Numerical training features.
        y_train: Binary defect labels (0 or 1).
        max_depth: Maximum number of tree levels.

    Returns:
        Fitted DecisionTreeRegressor model.
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def predict_defects(model, X, threshold=0.5):
    """Convert regression outputs to defect predictions.

    Args:
        model: A fitted regression model.
        X: Numerical image features.
        threshold: Decision threshold.

    Returns:
        Binary NumPy array of predictions.
    """
    return (
        model.predict(X) >= threshold
    ).astype(int)

