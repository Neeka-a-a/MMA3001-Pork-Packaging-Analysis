from src.preprocessing import (
    extract_image_features,
    extract_regional_features
)
"""Tests for image preprocessing."""

import cv2
import numpy as np
import pytest

from src.preprocessing import extract_image_features


def test_uniform_image(tmp_path):
    """Check features from a uniform grey image."""

    image = np.full(
        (100, 100, 3), 128, dtype=np.uint8
    )

    path = tmp_path / "test.png"
    cv2.imwrite(str(path), image)

    features = extract_image_features(path)

    assert len(features) == 4
    assert np.isclose(features[0], 128)
    assert np.isclose(features[1], 0)
    assert np.isclose(features[2], 0)
    assert np.isclose(features[3], 0)


def test_missing_image():
    """Check that a missing image raises an error."""

    with pytest.raises(ValueError):
        extract_image_features(
            "nonexistent_image_12345.png"
        )


def test_feature_values(tmp_path):
    """Check features contain valid numbers."""

    rng = np.random.default_rng(42)

    image = rng.integers(
        0, 256, (100, 100, 3), dtype=np.uint8
    )

    path = tmp_path / "random.png"
    cv2.imwrite(str(path), image)

    features = extract_image_features(path)

    assert len(features) == 4
    assert np.all(np.isfinite(features))
def test_regional_features(tmp_path):
    """Verify a 3 x 3 grid produces 36 features."""
    image = np.full(
        (300, 300, 3), 128, dtype=np.uint8
    )

    image_path = tmp_path / "regional_test.png"
    cv2.imwrite(str(image_path), image)

    features = extract_regional_features(
        image_path, grid_size=3
    )

    assert features.shape == (36,)
    assert np.all(np.isfinite(features))
    assert np.allclose(features[0::4], 128)
