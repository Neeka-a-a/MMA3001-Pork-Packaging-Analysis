
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



def extract_regional_features(image_path, grid_size=3):
    """Extract numerical features from a fixed image grid.

    Args:
        image_path: Path to the image.
        grid_size: Number of rows and columns.

    Returns:
        NumPy array of regional image features.
    """
    if grid_size < 1:
        raise ValueError("grid_size must be positive")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Cannot load image: {image_path}"
        )

    # Standardise image dimensions
    image = cv2.resize(image, (300, 300))

    features = []

    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * 300 // grid_size
            y2 = (row + 1) * 300 // grid_size
            x1 = col * 300 // grid_size
            x2 = (col + 1) * 300 // grid_size

            region = image[y1:y2, x1:x2]

            gray = cv2.cvtColor(
                region, cv2.COLOR_BGR2GRAY
            )

            hsv = cv2.cvtColor(
                region, cv2.COLOR_BGR2HSV
            )

            edges = cv2.Canny(gray, 50, 150)

            features.extend([
                np.mean(gray),
                np.std(gray),
                np.mean(hsv[:, :, 1]),
                np.mean(edges > 0)
            ])

    return np.asarray(features, dtype=float)

