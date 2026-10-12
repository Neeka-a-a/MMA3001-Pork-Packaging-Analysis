
"""Image preprocessing for the MMA3001 project."""

from pathlib import Path

import cv2
import numpy as np


def extract_image_features(image_path):
    """Extract numerical image characteristics.

    Args:
        image_path: Path to an RGB image file.

    Returns:
        Array containing mean brightness, contrast,
        mean saturation and edge density.

    Raises:
        ValueError: If the image cannot be loaded.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Cannot load image: {image_path}"
        )

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    )

    edges = cv2.Canny(gray, 50, 150)

    features = np.array([
        np.mean(gray),
        np.std(gray),
        np.mean(hsv[:, :, 1]),
        np.mean(edges > 0)
    ], dtype=float)

    return features
