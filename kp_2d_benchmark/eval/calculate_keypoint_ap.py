"""Average precision for the benchmark, using the metric of the keypoint-detection submodule.

This is a distance-threshold AP, not COCO OKS: a detection is a true positive when it lies within a
threshold distance of a ground truth keypoint of the same type. The matching, the precision-recall curve
and the AP integration all come from `keypoint_detection.models.metrics`, so the benchmark reports the same
metric that the heatmap models are trained and checkpointed against.

Two deliberate differences with how that metric is configured during training:

- The threshold is normalised. Keypoint coordinates are divided by the object scale before matching, so the
  threshold is an alpha rather than a pixel count and the number means the same thing on a 256x256 crop as
  on a 512x512 one. The metric itself is used unmodified; only its inputs are scaled.
- The result files hold a single detection per keypoint channel, whereas training extracts up to 20
  candidate peaks per channel. The AP values are therefore not comparable to the `test/meanAP` that the
  training loop logs. See the "Metrics" section of the README.

Out of view keypoints (v = 0) are *not* skipped here, unlike in the distance metrics: their ground truth is
empty, so a model that still predicts a keypoint there takes a false positive. This is what the metric does
at training time, and it is the only metric in the benchmark that rewards knowing a keypoint is absent.
"""

from typing import Dict, List

from keypoint_detection.models.metrics import DetectedKeypoint, Keypoint, KeypointAPMetric

from kp_2d_benchmark.eval.calculate_keypoint_distance_metrics import index_by_image_id, object_scale
from kp_2d_benchmark.eval.coco_results import COCOKeypointResults, CocoKeypointsDataset

AP_ALPHA = 0.05
"""The benchmark's AP threshold, as a fraction of the object scale. Shared with PCK@0.05 on purpose: both
metrics then call a keypoint correct under the same criterion, so the difference between the two columns is
exactly what the confidence ranking and the false positives add."""


def calculate_keypoint_ap(
    coco_dataset: CocoKeypointsDataset,
    coco_results: COCOKeypointResults,
    alpha: float = AP_ALPHA,
    normalize: bool = True,
) -> Dict[str, float]:
    """Average precision per keypoint channel, and their mean.

    Args:
        alpha: threshold distance. A fraction of the object scale if normalize is True, a pixel distance
            otherwise. Pixel thresholds are what the keypoint-detection training loop uses.
        normalize: divide keypoint coordinates by the object scale before matching.

    Returns:
        {"mAP": float, "<keypoint name>": float, ...}
    """
    image_id_to_annotation, image_id_to_prediction = index_by_image_id(coco_dataset, coco_results)

    # the aRTF datasets declare all four cloth categories but annotate only one of them, so an unused
    # category is normal and must not be evaluated. within an annotated category, a keypoint that is never
    # in view is a dataset problem rather than a valid empty channel.
    annotated_category_ids = {annotation.category_id for annotation in image_id_to_annotation.values()}

    ap_per_channel: Dict[str, float] = {}
    for category in coco_dataset.categories:
        if category.id not in annotated_category_ids:
            continue
        for channel_index, channel_name in enumerate(category.keypoints):
            metric = KeypointAPMetric(alpha)
            n_ground_truth = 0

            for image_id, annotation in image_id_to_annotation.items():
                if annotation.category_id != category.id:
                    continue
                scale = object_scale(annotation) if normalize else 1.0

                gt_keypoints = _channel_ground_truth(annotation, channel_index, scale)
                n_ground_truth += len(gt_keypoints)
                detected_keypoints = _channel_detections(image_id_to_prediction.get(image_id), channel_index, scale)

                # an image without a prediction must still be passed in, otherwise its ground truth is
                # never counted and the missed detection silently stops hurting recall.
                metric.update(detected_keypoints, gt_keypoints)

            if n_ground_truth == 0:
                raise ValueError(
                    f"Keypoint '{channel_name}' of category '{category.name}' is not in view in a single "
                    "image of this split, so its average precision is undefined."
                )
            ap_per_channel[channel_name] = float(metric.compute())

    if not ap_per_channel:
        raise ValueError("Cannot compute an average precision without any keypoint channels.")

    ap_per_channel["mAP"] = sum(ap_per_channel.values()) / len(ap_per_channel)
    return ap_per_channel


def _channel_ground_truth(annotation, channel_index: int, scale: float) -> List[Keypoint]:
    """The ground truth keypoints of one channel, empty if the keypoint is not in view.

    Keypoint annotates its coordinates as int, but nothing enforces that and the distance it computes is
    plain float arithmetic, so the scaled coordinates are fine.
    """
    u, v, visibility = annotation.keypoints[3 * channel_index : 3 * channel_index + 3]
    if visibility == 0:
        return []
    return [Keypoint(u / scale, v / scale)]


def _channel_detections(prediction, channel_index: int, scale: float) -> List[DetectedKeypoint]:
    """The predicted keypoints of one channel, empty if the model detected nothing in this image."""
    if prediction is None:
        return []
    u, v, _ = prediction.keypoints[3 * channel_index : 3 * channel_index + 3]
    if prediction.per_keypoint_scores is not None:
        probability = prediction.per_keypoint_scores[channel_index]
    else:
        # fall back to the instance score, which ranks all channels of an image identically
        probability = prediction.score
    return [DetectedKeypoint(u / scale, v / scale, probability)]
