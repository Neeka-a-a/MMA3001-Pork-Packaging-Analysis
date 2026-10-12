
"""Dataset loading utilities for COCO annotations."""

import json
from collections import defaultdict
from pathlib import Path


def load_coco_annotations(annotation_path):
    """Read a COCO JSON annotation file."""
    with open(annotation_path, "r") as file:
        data = json.load(file)

    required = {"images", "annotations", "categories"}

    if not required.issubset(data):
        raise ValueError("Invalid COCO annotation file")

    return data


def get_defect_labels(coco_data, target_class):
    """Get image-level binary labels for one defect.

    Args:
        coco_data: Parsed COCO annotations.
        target_class: Name of defect to detect.

    Returns:
        Dictionary mapping image IDs to 0 or 1.
    """
    category_ids = {
        category["id"]
        for category in coco_data["categories"]
        if category["name"] == target_class
    }

    if not category_ids:
        raise ValueError(
            f"Unknown defect class: {target_class}"
        )

    positive_image_ids = {
        annotation["image_id"]
        for annotation in coco_data["annotations"]
        if annotation["category_id"] in category_ids
    }

    return {
        image["id"]: int(
            image["id"] in positive_image_ids
        )
        for image in coco_data["images"]
    }


def create_image_records(
    image_folder, annotation_path, target_class
):
    """Combine image paths with defect-presence labels."""
    image_folder = Path(image_folder)

    coco_data = load_coco_annotations(
        annotation_path
    )

    labels = get_defect_labels(
        coco_data, target_class
    )

    records = []

    for image in coco_data["images"]:
        image_path = (
            image_folder / image["file_name"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        records.append({
            "image_path": str(image_path),
            "label": labels[image["id"]],
            "image_id": image["id"]
        })

    return records

