# 2D-keypoints-benchmark
A benchmark for 2D category-level keypoint detection with a strong focus on machine vision.


## Datasets

| | image resolution | # keypoints | # train images | # val images | # test images |
|---|---|---|---|---|---|
RoboFlow Garlic | 256x256 | 2 | 697 | 104 | 199 |
aRTF Towels | 512x256 | 4 | 168 | 42 | 400 |
aRTF Shorts | 512x256 | 7 | 83 | 21 | 180 |
aRTF Tshirts | 512x256 | 12 | 168 | 42 | 400 |
CUB-200-2011 | 512x512 | 15 | 5395 | 599 | 5794 |
AP-10K | 512x512 | 17 | 9122 | 1272 | 2634 |
GITW (Glasses-in-the-Wild) | 256x256 | 5 | 621 | 69 | 310 |

All datasets are stored in COCO keypoints format and are downloaded from huggingface by
`DatasetContainer.download()`. Two of them are repackaged from their upstream release:

- **AP-10K** declares 54 species categories that all share the same 17-keypoint quadruped skeleton, and
  contains ~1.3 annotated animals per image. The species are collapsed into a single `animal` category and
  every instance is cropped into its own image (square, bbox + 25% margin), see
  `kp_2d_benchmark/datasets/ap10k_to_coco.py`. This makes it a top-down task that assumes a perfect
  detector, which keeps it object-centric like the other datasets.
- **GITW** ships one 256x256 crop per glass. The published validation split is used here as the test split
  and a fresh validation split is carved out of the published train split, so that model selection never
  touches the reported test set. Bounding boxes are derived from the keypoints, since the published
  annotations contain none, see `kp_2d_benchmark/datasets/gitw_to_coco.py`.


## Scope and limitations

**Single instance per image.** Every dataset in the benchmark has at most one annotated instance per image,
and the evaluation code relies on this: `calculate_keypoint_distances` matches predictions to ground truth
by `image_id` alone and raises on a second annotation for the same image. Multi-instance evaluation is
deliberately out of scope for now. Adding it requires a ground-truth-to-prediction assignment step
(e.g. Hungarian matching on OKS) plus a policy for unmatched instances, and both `create_coco_results_file`
implementations would have to emit more than one instance per image. Any genuinely crowded dataset
(ApolloCar3D, OCHuman, ...) needs that work first.

**Distances are in raw pixels.** The raw-pixel distance columns are not normalised by object or image size,
so they are only directly comparable between datasets of equal resolution. The normalised metrics defined
below do not have this problem.

**Images without a prediction are skipped by the raw-pixel distances.** If a model detects nothing in an
image, that image does not contribute to the distance columns rather than being penalised. The other
metrics each state their own policy, and three of the five are detection-aware.


## Metrics

All metrics are computed against the test split from the COCO-format result files in `data/results/`, by
`kp_2d_benchmark/eval/calculate_all_metrics.py`. No metric requires anything beyond the predicted keypoints
and their confidences, so the whole table can be regenerated without retraining.

### Shared conventions

- **Object scale.** `s = max(bbox_width, bbox_height)` of the ground-truth annotation. Every dataset is
  required to carry bounding boxes (see the dataset invariants), so `s` is always defined. Normalising by
  `s` is what makes the metrics comparable across datasets of different resolution.
- **Visibility.** Keypoints annotated `v = 0` (not in view) are excluded from the first four metrics.
  `v = 1` (labelled but occluded) and `v = 2` (visible) both count. mAP is the exception and treats a
  prediction for an out-of-view keypoint as a false positive; see its entry below.
- **Single instance.** At most one ground-truth annotation and one predicted instance per image, matched by
  `image_id`. See the scope section above.
- **Missing predictions.** Each metric states its own policy. This is deliberate rather than an oversight:
  a localisation metric that conditions on detection and a task metric that penalises it answer different
  questions, and reporting the detection rate next to both keeps the conditioning visible.

### The five reported metrics

**Detection rate.** Percentage of test images for which the model emitted any prediction. Not a quality
measure on its own; it is the denominator context for everything else, and without it a model that only
reports the images it finds easy looks better than it is.

**Median NME** (normalised mean error). Median over all visible keypoints of `||p - g||₂ / s`. Median
rather than mean because the error distribution has a heavy tail, and a handful of gross failures otherwise
dominate the average. Computed over detected images only, so it must be read together with the detection
rate.

**PCK@0.05.** Percentage of visible keypoints whose prediction lies within `0.05 · s` of the ground truth.
Missing predictions count as incorrect, so this metric is detection-aware. `α = 0.05` is used because
`α = 0.1` saturates on most datasets in this benchmark and stops separating models; 0.1 is the more common
value in the literature and is easy to add as a second column if a comparison calls for it. Note that
AP-10K crops are the bounding box plus a 25% margin, so `s` is nearly constant there (~410 px) and
PCK@0.05 on that dataset is effectively a fixed 20 px threshold.

**Strict success@0.05.** Percentage of test images in which *every* visible keypoint is within `0.05 · s`.
Images without a prediction count as failures. This is the task-level view: per-keypoint PCK is optimistic
whenever the downstream application needs the whole keypoint configuration to be right, and the two can
differ by a lot on datasets with many keypoints.

**mAP.** Mean average precision as implemented in the pinned `keypoint-detection` submodule
(`keypoint_detection.models.metrics.KeypointAPMetrics`). This is a distance-threshold AP, not COCO OKS. A
detection counts as a true positive when it falls within a threshold distance of a ground-truth keypoint of
the same type; detections are matched greedily in order of confidence, unmatched ground truth counts as a
false negative and unmatched detections as false positives. AP is computed per keypoint channel over the
whole test split, then averaged over channels and over thresholds. Confidence scores enter through the
precision-recall ranking, which no other metric in the table uses.

Thresholds are normalised by dividing keypoint coordinates by `s` before matching, so a threshold is an `α`
rather than a pixel count, and `α ∈ {0.02, 0.05}` is used. This is the only deviation from the submodule's
own configuration, which uses raw pixel thresholds (default `2 4`); the matching and AP computation are
untouched.

The AP here is not numerically identical to the `test/meanAP` that the keypoint-detection training loop
logs, even at the same pixel thresholds. The result files store only the highest-scoring detection per
keypoint channel, whereas the training loop extracts up to 20 candidate peaks per channel, so the two
integrate different precision-recall curves. In this table the false positives can only come from a
mislocalised top-1 detection or from a prediction on an out-of-view keypoint, which makes the metric a
ranking over images rather than a full detection AP. Storing more than one candidate per channel would
require the result files to carry multiple instances per image, which the single-instance scope rules out.

Unlike the other four metrics, mAP does not ignore keypoints annotated `v = 0`: the ground truth for that
channel is empty, so a model that still emits a keypoint there takes a false positive. This is the
behaviour of the submodule metric at training time, and it makes mAP the only metric here that rewards
recognising that a keypoint is absent. It matters on AP-10K (41% of keypoint slots are out of view) and
CUB-200 (20%); on the other five datasets fewer than 0.2% of slots are affected.

Because the metric is the submodule's, the mAP column is tied to the submodule pin in the same way the
backbone results are.

The raw-pixel mean and median keypoint distance are kept in `metrics.csv` for continuity with earlier
versions of this table, but are no longer the headline numbers.


## Performance numbers

> The metrics defined above are specified but not yet implemented: the tables below still report the
> legacy raw-pixel distances. They are regenerated from the existing result files, without retraining,
> once the evaluation code lands.

**average keypoint distance**

| model          |   AP10K_512 |   ARTF_Shorts_Dataset |   ARTF_Towels_Dataset |   ARTF_Tshirts_Dataset |   CUB200_2011_512 |   GITW_256 |   RoboflowGarlic256Dataset |
|:---------------|------------:|----------------------:|----------------------:|-----------------------:|------------------:|-----------:|---------------------------:|
| pkd-DinoV2Up   |        34.9 |                  16.5 |                   9   |                   19.7 |               9.8 |        7.1 |                        4.8 |
| pkd-MaxVitUnet |        37.3 |                  40.4 |                  17.8 |                   17.8 |              11.3 |        5   |                        4.9 |
| yolo26s        |        28.9 |                  89.1 |                  21.8 |                   41.5 |              11.3 |        7.7 |                        4.1 |
| yolov8         |        34.5 |                  60   |                  20.9 |                   56.4 |              13.2 |       10.9 |                        5.9 |

**median keypoint distance**

| model          |   AP10K_512 |   ARTF_Shorts_Dataset |   ARTF_Towels_Dataset |   ARTF_Tshirts_Dataset |   CUB200_2011_512 |   GITW_256 |   RoboflowGarlic256Dataset |
|:---------------|------------:|----------------------:|----------------------:|-----------------------:|------------------:|-----------:|---------------------------:|
| pkd-DinoV2Up   |        12.2 |                   3.8 |                   1.8 |                    3.4 |               6.4 |        3   |                        3   |
| pkd-MaxVitUnet |        13   |                   7.7 |                   1.3 |                    2.8 |               6.7 |        2.1 |                        3.1 |
| yolo26s        |        15.2 |                  46.9 |                   7.3 |                   13.3 |               7.9 |        4.7 |                        3.2 |
| yolov8         |        20.2 |                  32.8 |                   8.7 |                   22.5 |               9.5 |        7.3 |                        4.9 |

Numbers are produced by `kp_2d_benchmark/eval/calculate_all_metrics.py` from the result files in
`data/results/`, and are reproducible: the training uses a fixed seed, and re-running a model on an
unchanged dataset reproduces its result file.

The yolo rows are `yolov8s-pose` and `yolo26s-pose`, both finetuned under the pinned ultralytics version.

### Two pinned dependencies

Both have already produced mislabeled numbers, so keep them pinned when updating results:

- The **`keypoint-detection` submodule**, because the backbone that produced an earlier version of these
  numbers was renamed and changed upstream while the parent repo pointed at a commit that did not contain
  it at all.
- **`ultralytics`**, because the version changes yolo results in both directions. Going from 8.3.58 to
  8.4.120 moved AP-10K mean distance from 87.6 to 35.6 and CUB from 26.0 to 13.5, but ARTF tshirts from
  26.9 to 55.0.

### Known weak spots in the table

- **ARTF_Shorts is poor for both yolo models** (median 32.8 and 46.9). It is the smallest training split in
  the benchmark, so at 100 epochs and batch 16 it gets only ~600 optimiser steps, against ~57,000 for
  AP-10K. The budget is specified in epochs, so it scales with dataset size; these cells are likely
  undertrained rather than genuinely hard.
- **`yolo26s` detects nothing in 21% of the ARTF_Tshirts test set** (315/400). Since images without a
  prediction are skipped, that cell is computed on the images the model did find.


## Local Development

### Local installation

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.
