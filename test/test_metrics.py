"""Tests for the benchmark metrics.

The fixture below is small enough to compute every expected value by hand, and covers the four situations
that the metrics have to disagree about: an exact prediction, a mislocalised one, an image the model did
not detect at all, and a keypoint that is out of view but predicted anyway.
"""

import copy

import pytest

from kp_2d_benchmark.eval.calculate_all_metrics import parse_result_file_name
from kp_2d_benchmark.eval.calculate_keypoint_ap import calculate_keypoint_ap
from kp_2d_benchmark.eval.calculate_keypoint_distance_metrics import (
    calculate_average_distances,
    calculate_detection_rate,
    calculate_keypoint_distances,
    calculate_median_normalized_distance,
    calculate_pck_at_alpha,
    calculate_strict_success_rate,
    object_scale,
)
from kp_2d_benchmark.eval.coco_results import COCOKeypointResults, CocoKeypointsDataset

# the bounding box is 100x50, so the object scale is 100 and an alpha of 0.05 is a 5 pixel threshold
BBOX = (0.0, 0.0, 100.0, 50.0)
SCALE = 100.0
ALPHA = 0.05

# heel, nose, top at (10,10), (20,20), (30,30), all in view
VISIBLE_KEYPOINTS = [10, 10, 2, 20, 20, 2, 30, 30, 2]
# same, but "top" is out of view
TOP_OUT_OF_VIEW_KEYPOINTS = [10, 10, 2, 20, 20, 2, 30, 30, 0]


def _image(image_id):
    return {"id": image_id, "width": 100, "height": 100, "file_name": f"{image_id}.jpg"}


def _annotation(image_id, keypoints):
    return {
        "id": image_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": BBOX,
        "keypoints": keypoints,
    }


@pytest.fixture
def dataset():
    return CocoKeypointsDataset(
        categories=[{"supercategory": "object", "id": 1, "name": "shoe", "keypoints": ["heel", "nose", "top"]}],
        images=[_image(i) for i in (1, 2, 3, 4)],
        annotations=[
            _annotation(1, VISIBLE_KEYPOINTS),  # predicted exactly
            _annotation(2, VISIBLE_KEYPOINTS),  # heel is 10 pixels off
            _annotation(3, VISIBLE_KEYPOINTS),  # not detected at all
            _annotation(4, TOP_OUT_OF_VIEW_KEYPOINTS),  # top is out of view but predicted anyway
        ],
    )


@pytest.fixture
def results():
    return COCOKeypointResults(
        [
            {
                "image_id": 1,
                "category_id": 1,
                "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 2],
                "score": 0.9,
                "per_keypoint_scores": [0.9, 0.9, 0.9],
            },
            {
                "image_id": 2,
                "category_id": 1,
                "keypoints": [20, 10, 2, 20, 20, 2, 30, 30, 2],  # heel is 10 pixels off
                "score": 0.7,
                "per_keypoint_scores": [0.5, 0.8, 0.8],
            },
            # image 3 is missing on purpose
            {
                "image_id": 4,
                "category_id": 1,
                "keypoints": [10, 10, 2, 20, 20, 2, 30, 30, 2],  # top is predicted although it is not in view
                "score": 0.8,
                "per_keypoint_scores": [0.7, 0.7, 0.95],
            },
        ]
    )


@pytest.fixture
def distance_dict(dataset, results):
    return calculate_keypoint_distances(dataset, results)


def test_object_scale_is_the_longest_bbox_side(dataset):
    assert object_scale(dataset.annotations[0]) == SCALE


def test_object_scale_requires_a_bounding_box(dataset):
    annotation = copy.deepcopy(dataset.annotations[0])
    annotation.bbox = None
    with pytest.raises(ValueError, match="no bounding box"):
        object_scale(annotation)


def test_multiple_annotations_per_image_are_rejected(dataset, results):
    dataset.annotations = list(dataset.annotations) + [_annotation_model(dataset, 1)]
    with pytest.raises(ValueError, match="multiple annotations"):
        calculate_keypoint_distances(dataset, results)


def _annotation_model(dataset, image_id):
    duplicate = copy.deepcopy(dataset.annotations[0])
    duplicate.image_id = image_id
    return duplicate


def test_undetected_images_do_not_affect_the_raw_pixel_distances(distance_dict):
    """The legacy columns skip undetected images, which is what keeps them backwards compatible."""
    average = calculate_average_distances(distance_dict)[1]
    # heel: 0 in image 1, 10 in image 2, 0 in image 4. image 3 was not detected and is skipped.
    assert average["heel"] == pytest.approx(10 / 3)
    assert average["nose"] == pytest.approx(0.0)
    # top is out of view in image 4, so only images 1 and 2 contribute
    assert average["top"] == pytest.approx(0.0)


def test_detection_rate(distance_dict):
    # 3 of the 4 images have a prediction
    assert calculate_detection_rate(distance_dict) == pytest.approx(0.75)


def test_median_normalized_distance(distance_dict):
    # the predicted normalised distances are seven zeroes and one 0.1
    assert calculate_median_normalized_distance(distance_dict) == pytest.approx(0.0)


def test_pck_counts_undetected_images_as_incorrect(distance_dict):
    # 11 visible keypoints, of which 7 are within 5 pixels: image 3 contributes 3 failures and the
    # mislocalised heel of image 2 one more.
    assert calculate_pck_at_alpha(distance_dict, alpha=ALPHA) == pytest.approx(7 / 11)


def test_strict_success_rate_is_per_image(distance_dict):
    # image 1 and image 4 are fully correct, image 2 has a bad heel and image 3 was not detected
    assert calculate_strict_success_rate(distance_dict, alpha=ALPHA) == pytest.approx(0.5)


def test_strict_success_rate_is_at_most_pck(distance_dict):
    pck = calculate_pck_at_alpha(distance_dict, alpha=ALPHA)
    assert calculate_strict_success_rate(distance_dict, alpha=ALPHA) <= pck


def test_average_precision_matches_hand_computed_values(dataset, results):
    ap = calculate_keypoint_ap(dataset, results, alpha=ALPHA)

    # heel: 4 ground truth keypoints, detections ranked 0.9 (TP), 0.7 (TP), 0.5 (FP, 10 pixels off).
    assert ap["heel"] == pytest.approx(0.5)
    # nose: 4 ground truth keypoints and 3 true positives, so recall tops out at 0.75 at full precision.
    assert ap["nose"] == pytest.approx(0.75)
    # top: 3 ground truth keypoints, and the highest scoring detection (0.95) is the out of view false
    # positive of image 4, which depresses precision at every recall level.
    assert ap["top"] == pytest.approx(4 / 9)
    assert ap["mAP"] == pytest.approx((0.5 + 0.75 + 4 / 9) / 3)


def test_average_precision_penalises_predicting_an_out_of_view_keypoint(dataset, results):
    with_false_positive = calculate_keypoint_ap(dataset, results, alpha=ALPHA)

    # drop the prediction for image 4 entirely, so the out of view keypoint is no longer predicted
    without_image_4 = COCOKeypointResults([r for r in results if r.image_id != 4])
    without_false_positive = calculate_keypoint_ap(dataset, without_image_4, alpha=ALPHA)

    assert without_false_positive["top"] > with_false_positive["top"]


def test_average_precision_charges_for_a_missed_detection(dataset, results):
    """Removing the undetected image from the ground truth can only raise AP, because it removes a FN."""
    with_missed_detection = calculate_keypoint_ap(dataset, results, alpha=ALPHA)

    detected_only = copy.deepcopy(dataset)
    detected_only.images = [image for image in detected_only.images if image.id != 3]
    detected_only.annotations = [a for a in detected_only.annotations if a.image_id != 3]
    without_missed_detection = calculate_keypoint_ap(detected_only, results, alpha=ALPHA)

    assert without_missed_detection["mAP"] > with_missed_detection["mAP"]


def test_average_precision_rejects_a_never_visible_keypoint(dataset, results):
    for annotation in dataset.annotations:
        annotation.keypoints[8] = 0  # "top" is out of view everywhere
    with pytest.raises(ValueError, match="not in view in a single image"):
        calculate_keypoint_ap(dataset, results, alpha=ALPHA)


def test_normalisation_makes_alpha_scale_invariant(dataset, results):
    """Doubling the object and its bounding box leaves every normalised metric unchanged."""
    ap = calculate_keypoint_ap(dataset, results, alpha=ALPHA)
    pck = calculate_pck_at_alpha(calculate_keypoint_distances(dataset, results), alpha=ALPHA)

    scaled_dataset = copy.deepcopy(dataset)
    for annotation in scaled_dataset.annotations:
        annotation.bbox = tuple(2 * v for v in annotation.bbox)
        annotation.keypoints = [v if i % 3 == 2 else 2 * v for i, v in enumerate(annotation.keypoints)]
    scaled_results = COCOKeypointResults(
        [
            r.model_copy(update={"keypoints": [v if i % 3 == 2 else 2 * v for i, v in enumerate(r.keypoints)]})
            for r in results
        ]
    )

    assert calculate_keypoint_ap(scaled_dataset, scaled_results, alpha=ALPHA)["mAP"] == pytest.approx(ap["mAP"])
    assert calculate_pck_at_alpha(
        calculate_keypoint_distances(scaled_dataset, scaled_results), alpha=ALPHA
    ) == pytest.approx(pck)


@pytest.mark.parametrize(
    "file_name, model, dataset_name",
    [
        ("model=yolov8,dataset=GITW_256.json", "yolov8", "GITW_256"),
        ("/a/b/model=pkd-DinoV2Up,dataset=AP10K_512.json", "pkd-DinoV2Up", "AP10K_512"),
    ],
)
def test_parse_result_file_name(file_name, model, dataset_name):
    assert parse_result_file_name(file_name) == (model, dataset_name)
