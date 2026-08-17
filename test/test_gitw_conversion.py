"""Tests for the GITW split rebuild: the published val split becomes our test split, a new val
split is carved out of the published train split, and bounding boxes are derived from the keypoints."""

import json

import numpy as np
import pytest
from PIL import Image

from kp_2d_benchmark.datasets.gitw_to_coco import VAL_FRACTION, bbox_from_keypoints, write_split
from kp_2d_benchmark.eval.coco_results import CocoKeypointsDataset

KEYPOINT_NAMES = ["bottom_front", "top_left", "top_right", "top_front", "fluid_level"]


def make_published_split(tmp_path, name, num_images, first_id):
    """Mimic a published split: images plus a coco file that lives inside the images folder."""
    image_dir = tmp_path / name / "images"
    image_dir.mkdir(parents=True)

    images, annotations = [], []
    for i in range(num_images):
        file_name = f"anon{i}_id{first_id + i}_glass0.png"
        Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(image_dir / file_name)
        images.append({"id": first_id + i, "width": 256, "height": 256, "file_name": file_name})
        annotations.append(
            {
                "id": i + 1,
                "image_id": first_id + i,
                "category_id": 1,
                "keypoints": [100.0, 200.0, 2, 60.0, 70.0, 2, 200.0, 100.0, 2, 120.0, 140.0, 2, 110.0, 170.0, 2],
                "num_keypoints": 5,
            }
        )

    raw = {
        "categories": [{"supercategory": "", "id": 1, "name": "glass", "keypoints": KEYPOINT_NAMES}],
        "images": images,
        "annotations": annotations,
    }
    (image_dir / "annotations.json").write_text(json.dumps(raw))
    return raw, image_dir


def test_bbox_wraps_the_keypoints_with_margin():
    keypoints = [100.0, 100.0, 2, 200.0, 180.0, 2]
    x, y, w, h = bbox_from_keypoints(keypoints, image_width=256, image_height=256)
    assert x < 100 and y < 100, "box must be padded outwards"
    assert x + w > 200 and y + h > 180
    assert x >= 0 and y >= 0 and x + w <= 256 and y + h <= 256


def test_bbox_is_clipped_to_the_image():
    # keypoints hard against the image edges, the margin must not push the box outside
    keypoints = [0.0, 0.0, 2, 256.0, 256.0, 2]
    x, y, w, h = bbox_from_keypoints(keypoints, image_width=256, image_height=256)
    assert (x, y) == (0, 0)
    assert x + w <= 256 and y + h <= 256


def test_bbox_ignores_unlabeled_keypoints():
    # the unlabeled (0,0,0) keypoint must not drag the box to the origin
    keypoints = [100.0, 100.0, 2, 200.0, 180.0, 2, 0.0, 0.0, 0]
    x, y, _, _ = bbox_from_keypoints(keypoints, image_width=256, image_height=256)
    assert x > 50 and y > 50


def test_bbox_requires_a_labeled_keypoint():
    with pytest.raises(AssertionError):
        bbox_from_keypoints([0.0, 0.0, 0], image_width=256, image_height=256)


def test_write_split_produces_a_valid_dataset_with_boxes(tmp_path):
    raw, image_dir = make_published_split(tmp_path, "published_train", num_images=10, first_id=1)
    target = tmp_path / "out"

    write_split(raw, image_dir, [image["id"] for image in raw["images"]][:4], target)

    dataset = CocoKeypointsDataset(**json.loads((target / "annotations.json").read_text()))
    assert len(dataset.images) == 4
    assert len(dataset.annotations) == 4
    # images are copied into an images/ subfolder, like the other datasets
    for image in dataset.images:
        assert image.file_name.startswith("images/")
        assert (target / image.file_name).exists()
    # every annotation gained a bbox, which the yolo converter needs
    assert all(annotation.bbox is not None for annotation in dataset.annotations)
    assert list(dataset.categories[0].keypoints) == KEYPOINT_NAMES


def test_convert_uses_published_val_as_test_and_carves_val_out_of_train(tmp_path, monkeypatch):
    import kp_2d_benchmark.datasets.gitw_to_coco as m

    raw_dir = tmp_path / "raw"
    make_published_split(raw_dir, "glasses_in_the_wild_1000_train", num_images=100, first_id=1)
    make_published_split(raw_dir, "glasses_in_the_wild_1000_val", num_images=30, first_id=1000)

    monkeypatch.setattr(m, "GITW_RAW_DIR", raw_dir)
    monkeypatch.setattr(m, "GITW_256_DIR", tmp_path / "out")
    m.convert()

    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = CocoKeypointsDataset(**json.loads((tmp_path / "out" / split / "annotations.json").read_text()))

    # the published val split is the test split, untouched
    assert len(splits["test"].images) == 30
    # train and val partition the published train split, without overlap
    assert len(splits["val"].images) == round(100 * VAL_FRACTION)
    assert len(splits["train"].images) == 100 - round(100 * VAL_FRACTION)
    train_ids = {image.id for image in splits["train"].images}
    val_ids = {image.id for image in splits["val"].images}
    test_ids = {image.id for image in splits["test"].images}
    assert not train_ids & val_ids
    assert not (train_ids | val_ids) & test_ids
