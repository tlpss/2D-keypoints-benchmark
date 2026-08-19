"""Tests for the AP-10K instance cropping, which has to turn a multi-instance, 54-category dataset
into the single-instance, single-category format that the rest of the benchmark assumes."""

import json

import numpy as np
from PIL import Image

from kp_2d_benchmark.datasets.ap10k_to_coco import (
    CATEGORY_ID,
    CATEGORY_NAME,
    compute_square_crop_box,
    convert_split,
    crop_keypoints,
)
from kp_2d_benchmark.eval.coco_results import CocoKeypointsDataset

KEYPOINT_NAMES = [f"kp{i}" for i in range(17)]


def make_raw_dataset(tmp_path):
    """A miniature AP-10K: two species categories, one single-animal image and one two-animal image."""
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    for name, (width, height) in {"a.jpg": (200, 100), "b.jpg": (200, 100)}.items():
        Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(image_dir / name)

    def keypoints(x, y):
        # all 17 keypoints stacked on one location, visible
        return [x, y, 2] * 17

    raw = {
        "categories": [
            {"id": 1, "name": "dog", "supercategory": "Canidae", "keypoints": KEYPOINT_NAMES, "skeleton": [[1, 2]]},
            {"id": 2, "name": "cat", "supercategory": "Felidae", "keypoints": KEYPOINT_NAMES, "skeleton": [[1, 2]]},
        ],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 200, "height": 100},
            {"id": 2, "file_name": "b.jpg", "width": 200, "height": 100},
        ],
        "annotations": [
            # single animal, comfortably inside the image
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [80, 40, 40, 20], "keypoints": keypoints(100, 50)},
            # two animals in one image, of different species
            {"id": 11, "image_id": 2, "category_id": 1, "bbox": [10, 10, 40, 40], "keypoints": keypoints(30, 30)},
            {"id": 12, "image_id": 2, "category_id": 2, "bbox": [150, 50, 40, 40], "keypoints": keypoints(170, 70)},
        ],
    }
    raw_json_path = tmp_path / "raw-split1.json"
    raw_json_path.write_text(json.dumps(raw))
    return raw_json_path, image_dir


def test_crop_box_is_square_and_has_margin():
    x0, y0, x1, y1 = compute_square_crop_box([80, 40, 40, 20], image_width=200, image_height=100)
    assert x1 - x0 == y1 - y0, "crop must be square so the resize does not distort"
    assert x1 - x0 == round(1.25 * 40)
    # centred on the bbox centre
    assert (x0 + x1) / 2 == 100
    assert (y0 + y1) / 2 == 50


def test_crop_box_is_shifted_inside_the_image_instead_of_hanging_over_the_edge():
    # bbox in the top left corner, the margin would push the box to negative coordinates
    x0, y0, x1, y1 = compute_square_crop_box([0, 0, 40, 40], image_width=200, image_height=100)
    assert (x0, y0) == (0, 0)
    assert x1 - x0 == y1 - y0 == 50


def test_crop_box_stays_square_when_the_bbox_is_larger_than_the_image():
    # a 100px square + margin does not fit in a 100px high image, so the box must hang over and be padded
    x0, y0, x1, y1 = compute_square_crop_box([50, 0, 100, 100], image_width=200, image_height=100)
    assert x1 - x0 == y1 - y0 == 125
    assert y0 < 0 or y1 > 100


def test_keypoints_outside_the_crop_are_marked_unlabeled():
    # one keypoint inside the crop, one outside it
    keypoints = [15.0, 15.0, 2] + [500.0, 500.0, 2]
    cropped = crop_keypoints(keypoints, x0=10, y0=10, crop_width=20, crop_height=20)
    assert cropped[:3] == [5.0, 5.0, 2]
    assert cropped[3:] == [0.0, 0.0, 0]


def test_already_unlabeled_keypoints_stay_unlabeled():
    cropped = crop_keypoints([15.0, 15.0, 0], x0=10, y0=10, crop_width=20, crop_height=20)
    assert cropped == [0.0, 0.0, 0]


def test_conversion_yields_one_instance_per_image_and_a_single_category(tmp_path):
    raw_json_path, image_dir = make_raw_dataset(tmp_path)
    target_dir = tmp_path / "cropped"

    convert_split(raw_json_path, image_dir, target_dir)

    # the result must parse as a coco keypoints dataset, which validates num_keypoints and the category ids
    dataset = CocoKeypointsDataset(**json.loads((target_dir / "annotations.json").read_text()))

    # the three instances of the two source images each became their own image
    assert len(dataset.images) == 3
    assert len(dataset.annotations) == 3
    assert len({annotation.image_id for annotation in dataset.annotations}) == 3

    # the 54 species categories (2 here) collapsed into one
    assert len(dataset.categories) == 1
    assert dataset.categories[0].name == CATEGORY_NAME
    assert list(dataset.categories[0].keypoints) == KEYPOINT_NAMES
    assert all(annotation.category_id == CATEGORY_ID for annotation in dataset.annotations)

    # every crop is square and was actually written to disk
    for image in dataset.images:
        assert image.width == image.height
        with Image.open(target_dir / image.file_name) as written:
            assert written.size == (image.width, image.height)

    # keypoints survived the translation into the crop frame
    for annotation in dataset.annotations:
        assert annotation.num_keypoints == 17
        for i in range(0, len(annotation.keypoints), 3):
            x, y, visibility = annotation.keypoints[i : i + 3]
            assert visibility == 2
            assert 0 <= x < 200 and 0 <= y < 200
