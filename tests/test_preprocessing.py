
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
