"""Keypoint metrics for the benchmark.

These metrics are single instance only: predictions are matched to ground truth by image_id, and an image
with more than one annotation is rejected rather than handled. Multi-instance evaluation is out of scope
for now, see the "Scope and limitations" section of the README. Supporting it needs a ground truth to
prediction assignment step (e.g. Hungarian matching on OKS) and a policy for unmatched instances, and both
create_coco_results_file implementations would have to emit more than one instance per image.

Every dataset in DATASETS satisfies the single instance assumption, by construction where needed: AP-10K is
cropped per instance and GITW is published as one crop per glass.

calculate_keypoint_distances returns KeypointError records rather than bare distances. Each record carries
the object scale it should be normalised by and whether the model predicted anything for that image at all,
which is what lets the detection-aware metrics (PCK, strict success rate) charge for a missed detection
while the raw pixel distances keep ignoring it. See the "Metrics" section of the README.
"""

import math
from typing import Dict, List, NamedTuple

from kp_2d_benchmark.eval.coco_results import COCOKeypointResults, CocoKeypointsDataset


class KeypointError(NamedTuple):
    """The error on a single ground truth keypoint.

    distance is in pixels and is math.inf if the model did not predict anything for this image, so that any
    threshold comparison counts it as a failure. scale is the object size used to normalise the distance.
    """

    distance: float
    scale: float
    image_id: int
    predicted: bool

    @property
    def normalized_distance(self) -> float:
        return self.distance / self.scale


# a distance dict maps category id -> keypoint name -> the errors on that keypoint over the whole split
DistanceDict = Dict[int, Dict[str, List[KeypointError]]]


def index_by_image_id(coco_dataset: CocoKeypointsDataset, coco_results: COCOKeypointResults):
    """Index the annotations and the predictions by image id, rejecting more than one of either.

    This is a hard requirement, not a sanity check: without instance matching there is no way to tell which
    annotation a prediction belongs to. See the module docstring.
    """
    image_id_to_annotation = {}
    for annotation in coco_dataset.annotations:
        if annotation.image_id in image_id_to_annotation:
            raise ValueError(
                f"Image {annotation.image_id} has multiple annotations, but these metrics are single "
                "instance only. Multi-instance evaluation is out of scope, see the module docstring."
            )
        image_id_to_annotation[annotation.image_id] = annotation

    image_id_to_prediction = {}
    for prediction in coco_results:
        if prediction.image_id in image_id_to_prediction:
            raise ValueError(
                f"Image {prediction.image_id} has multiple predictions, but these metrics are single "
                "instance only. Multi-instance evaluation is out of scope, see the module docstring."
            )
        image_id_to_prediction[prediction.image_id] = prediction

    return image_id_to_annotation, image_id_to_prediction


def object_scale(annotation) -> float:
    """The normalisation scale of an annotation: the longest side of its bounding box.

    Every dataset in the benchmark is required to carry bounding boxes, see the dataset invariants in
    AGENTS.md, but nothing enforced that at evaluation time until now.
    """
    if annotation.bbox is None:
        raise ValueError(
            f"Annotation {annotation.id} has no bounding box, so its keypoint errors cannot be normalised. "
            "Every dataset in the benchmark must provide bounding boxes, see the dataset invariants."
        )
    _, _, width, height = annotation.bbox
    scale = max(width, height)
    if scale <= 0:
        raise ValueError(f"Annotation {annotation.id} has a degenerate bounding box {annotation.bbox}.")
    return scale


def calculate_keypoint_distances(  # noqa: C901
    coco_dataset: CocoKeypointsDataset, coco_results: COCOKeypointResults
) -> DistanceDict:

    # create dict with categories and keypoint ids

    distance_dict: DistanceDict = {}
    for category in coco_dataset.categories:
        keypoint_dict: Dict[str, List[KeypointError]] = {}
        for i, keypoint in enumerate(category.keypoints):
            keypoint_dict[keypoint] = []
        distance_dict[category.id] = keypoint_dict

    category_id_to_category = {category.id: category for category in coco_dataset.categories}

    image_id_to_annotations, image_id_to_predictions = index_by_image_id(coco_dataset, coco_results)

    # then for each annotation, find the prediction

    for image_id, annotation in image_id_to_annotations.items():
        scale = object_scale(annotation)
        category_id = annotation.category_id
        annotated_keypoints = annotation.keypoints
        annotated_keypoints = [annotated_keypoints[i : i + 3] for i in range(0, len(annotated_keypoints), 3)]

        prediction = image_id_to_predictions.get(image_id)
        if prediction is None:
            # the model detected nothing in this image. record an infinite error for every keypoint that
            # should have been found, so that the detection-aware metrics can charge for it. the raw pixel
            # distances filter these out again, which keeps them backwards compatible.
            for i, name in enumerate(category_id_to_category[category_id].keypoints):
                if annotated_keypoints[i][2] == 0:
                    continue
                distance_dict[category_id][name].append(KeypointError(math.inf, scale, image_id, False))
            continue

        predicted_keypoints = prediction.keypoints
        predicted_keypoints = [predicted_keypoints[i : i + 3] for i in range(0, len(predicted_keypoints), 3)]

        for i, name in enumerate(category_id_to_category[category_id].keypoints):
            # calculate the distance between the predicted and annotated keypoint
            # add the distance to the list
            predicted_keypoint = predicted_keypoints[i]
            annotated_keypoint = annotated_keypoints[i]
            if annotated_keypoint[2] == 0:  # skip keypoints that are not in view.
                continue
            distance = (
                (predicted_keypoint[0] - annotated_keypoint[0]) ** 2
                + (predicted_keypoint[1] - annotated_keypoint[1]) ** 2
            ) ** 0.5
            distance_dict[category_id][name].append(KeypointError(distance, scale, image_id, True))

    # if there are remaining predictions, for which there was no annotation,
    # TODO: what to to with these? cannot take FP into account in distance metric?.
    # for now, just ignore them.

    # if there are categories for which we have not a single prediction, we should also ignore these
    # categories. note that this has to look at the predicted errors only, since every category now has
    # entries for the images that were not detected.
    new_distance_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        if all(not any(error.predicted for error in errors) for errors in keypoint_dict.values()):
            continue
        new_distance_dict[category_id] = keypoint_dict
    return new_distance_dict


def _predicted_distances(errors: List[KeypointError]) -> List[float]:
    """The pixel distances of the keypoints that were actually predicted.

    Images without a prediction are dropped here, which is what keeps the raw pixel distance metrics
    identical to the ones that were reported before the detection-aware metrics were added.
    """
    return [error.distance for error in errors if error.predicted]


def calculate_average_distances(distance_dict: DistanceDict):
    # calculate the average distance for each keypoint
    average_distance_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        average_distance_dict[category_id] = {}
        for keypoint_id, errors in keypoint_dict.items():
            distances = _predicted_distances(errors)
            average_distance_dict[category_id][keypoint_id] = sum(distances) / len(distances)
    return average_distance_dict


def calculate_median_distances(distance_dict: DistanceDict):
    # calculate the average distance for each keypoint
    median_distance_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        median_distance_dict[category_id] = {}
        for keypoint_id, errors in keypoint_dict.items():
            distances = _predicted_distances(errors)
            median_distance_dict[category_id][keypoint_id] = sorted(distances)[len(distances) // 2]

    return median_distance_dict


def calculate_std_deviation(distance_dict: DistanceDict):
    # calculate the average distance for each keypoint
    std_deviation_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        std_deviation_dict[category_id] = {}
        for keypoint_id, errors in keypoint_dict.items():
            distances = _predicted_distances(errors)
            mean = sum(distances) / len(distances)
            std_deviation_dict[category_id][keypoint_id] = (
                sum([(distance - mean) ** 2 for distance in distances]) / len(distances)
            ) ** 0.5
    return std_deviation_dict


def calculate_pck(distance_dict: DistanceDict, pixel_threshold=4):
    """
    Calculate the percentage of correct keypoints for a given pixel threshold.
    """
    pck_dict = {}
    for category_id, keypoint_dict in distance_dict.items():
        pck_dict[category_id] = {}
        for keypoint_id, errors in keypoint_dict.items():
            distances = _predicted_distances(errors)
            pck_dict[category_id][keypoint_id] = sum(1 for distance in distances if distance < pixel_threshold) / len(
                distances
            )
    return pck_dict


def _all_errors(distance_dict: DistanceDict) -> List[KeypointError]:
    return [error for keypoint_dict in distance_dict.values() for errors in keypoint_dict.values() for error in errors]


def calculate_detection_rate(distance_dict: DistanceDict) -> float:
    """Fraction of images for which the model predicted anything at all.

    This is the denominator context for every other metric: a model that only reports the images it finds
    easy looks better than it is on the metrics that skip undetected images.
    """
    detected_per_image = {}
    for error in _all_errors(distance_dict):
        # all errors of one image share the same prediction, so any of them carries the answer
        detected_per_image[error.image_id] = error.predicted
    if not detected_per_image:
        raise ValueError("Cannot compute a detection rate without any annotated keypoints.")
    return sum(detected_per_image.values()) / len(detected_per_image)


def calculate_median_normalized_distance(distance_dict: DistanceDict) -> float:
    """Median over all predicted keypoints of the distance normalised by the object scale (median NME).

    The median rather than the mean because the error distribution has a heavy tail, and undetected images
    are excluded because they have no distance to speak of, so this must be read next to the detection rate.
    """
    normalized = sorted(error.normalized_distance for error in _all_errors(distance_dict) if error.predicted)
    if not normalized:
        raise ValueError("Cannot compute a normalised distance without any predicted keypoints.")
    return normalized[len(normalized) // 2]


def calculate_pck_at_alpha(distance_dict: DistanceDict, alpha: float = 0.05) -> float:
    """Fraction of visible keypoints predicted within alpha * object scale of the ground truth.

    Undetected images have an infinite distance and hence count as incorrect, which makes this metric
    detection-aware.
    """
    errors = _all_errors(distance_dict)
    if not errors:
        raise ValueError("Cannot compute PCK without any annotated keypoints.")
    return sum(1 for error in errors if error.normalized_distance < alpha) / len(errors)


def calculate_strict_success_rate(distance_dict: DistanceDict, alpha: float = 0.05) -> float:
    """Fraction of images in which *every* visible keypoint is within alpha * object scale.

    The task level view: per keypoint PCK is optimistic whenever the downstream application needs the whole
    keypoint configuration to be right. Undetected images count as failures.
    """
    success_per_image: Dict[int, bool] = {}
    for error in _all_errors(distance_dict):
        correct = error.normalized_distance < alpha
        success_per_image[error.image_id] = success_per_image.get(error.image_id, True) and correct
    if not success_per_image:
        raise ValueError("Cannot compute a success rate without any annotated keypoints.")
    return sum(success_per_image.values()) / len(success_per_image)


if __name__ == "__main__":
    dataset_path = "/home/tlips/Code/2D-keypoints-benchmark/test/data/dummy_keypoints.json"
    coco_results_path = "/home/tlips/Code/2D-keypoints-benchmark/test/data/dummy_keypoint_results.json"
    import json

    with open(dataset_path, "r") as f:
        coco_dataset = CocoKeypointsDataset(**json.load(f))
    with open(coco_results_path, "r") as f:
        coco_results = COCOKeypointResults(json.load(f))
    distance_dict = calculate_keypoint_distances(coco_dataset, coco_results)
    average_distance_dict = calculate_average_distances(distance_dict)
    median_distance_dict = calculate_median_distances(distance_dict)
    std_deviation_dict = calculate_std_deviation(distance_dict)
    pck_dict = calculate_pck(distance_dict, pixel_threshold=4)
    print(pck_dict)
    print(average_distance_dict)
    print(f"detection rate: {calculate_detection_rate(distance_dict)}")
    print(f"median NME: {calculate_median_normalized_distance(distance_dict)}")
    print(f"PCK@0.05: {calculate_pck_at_alpha(distance_dict)}")
    print(f"strict success@0.05: {calculate_strict_success_rate(distance_dict)}")
