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
(`keypoint_detection.models.metrics.KeypointAPMetric`). This is a distance-threshold AP, not COCO OKS. A
detection counts as a true positive when it falls within a threshold distance of a ground-truth keypoint of
the same type; detections are matched greedily in order of confidence, unmatched ground truth counts as a
false negative and unmatched detections as false positives. AP is computed per keypoint channel over the
whole test split and then averaged over the channels, which is the mean in mAP. Confidence scores enter
through the precision-recall ranking, which no other metric in the table uses.

The threshold is normalised by dividing keypoint coordinates by `s` before matching, so it is an `α` rather
than a pixel count, and a single `α = 0.05` is used. This is the only deviation from the submodule's own
configuration, which uses raw pixel thresholds (default `2 4`); the matching and AP computation are
untouched.

Sharing `α` with PCK@0.05 is deliberate: both metrics call a keypoint correct under exactly the same
criterion, so the pair isolates what AP adds. PCK is the unweighted fraction of correct keypoints, while
mAP weights the same events by confidence rank and charges for false positives. A model whose confidences
order its predictions well scores higher on mAP than its PCK alone would suggest, and vice versa.

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

**Bold** marks the best result per dataset: the highest value, except for median NME where lower is
better. Column labels abbreviate the dataset names of the table at the top of this README.

**detection_rate**

| model          |    AP-10K |    Shorts |    Towels |   Tshirts |   CUB-200 |      GITW |    Garlic |
|:---------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| pkd-DinoV2Up   | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| pkd-MaxVitUnet | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| yolo26s        |      99.5 |      95.6 |      92.5 |      78.8 |      99.5 |      99.4 |      99.5 |
| yolov8         |      99.9 |      98.9 |      94.5 |      98.2 |      99.8 | **100.0** | **100.0** |

**median_nme**

| model          |    AP-10K |    Shorts |    Towels |   Tshirts |   CUB-200 |      GITW |    Garlic |
|:---------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| pkd-DinoV2Up   | **0.024** | **0.019** |     0.010 |     0.015 | **0.018** |     0.014 |     0.027 |
| pkd-MaxVitUnet |     0.026 |     0.026 | **0.007** | **0.013** | **0.018** | **0.011** | **0.026** |
| yolo26s        |     0.032 |     0.287 |     0.041 |     0.065 |     0.021 |     0.021 |     0.030 |
| yolov8         |     0.044 |     0.186 |     0.048 |     0.084 |     0.027 |     0.036 |     0.042 |

**pck@0.05**

| model          |   AP-10K |   Shorts |   Towels |   Tshirts |   CUB-200 |     GITW |   Garlic |
|:---------------|---------:|---------:|---------:|----------:|----------:|---------:|---------:|
| pkd-DinoV2Up   | **68.0** | **71.4** | **88.7** |  **78.2** |  **87.8** |     88.8 | **81.4** |
| pkd-MaxVitUnet |     65.0 |     54.6 |     79.3 |      77.6 |      84.9 | **92.1** |     78.4 |
| yolo26s        |     63.8 |      7.4 |     56.3 |      30.3 |      83.4 |     83.1 | **81.4** |
| yolov8         |     54.8 |     10.2 |     49.6 |      30.1 |      79.5 |     65.3 |     58.0 |

**strict_success@0.05**

| model          |   AP-10K |   Shorts |   Towels |   Tshirts |   CUB-200 |     GITW |   Garlic |
|:---------------|---------:|---------:|---------:|----------:|----------:|---------:|---------:|
| pkd-DinoV2Up   |     11.4 | **19.4** | **73.8** |  **24.8** |  **24.7** |     60.0 | **65.8** |
| pkd-MaxVitUnet |      9.4 |     11.7 |     50.7 |      22.2 |      18.7 | **74.5** |     58.3 |
| yolo26s        | **12.8** |      0.0 |     26.2 |       0.2 |      17.9 |     46.8 |     65.3 |
| yolov8         |      6.9 |      0.0 |     14.5 |       0.2 |      12.3 |     18.4 |     36.7 |

**mAP@0.05**

| model          |    AP-10K |    Shorts |    Towels |   Tshirts |   CUB-200 |      GITW |    Garlic |
|:---------------|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| pkd-DinoV2Up   | **0.559** | **0.627** | **0.850** | **0.728** | **0.790** |     0.867 |     0.717 |
| pkd-MaxVitUnet |     0.537 |     0.439 |     0.699 |     0.726 |     0.760 | **0.904** |     0.694 |
| yolo26s        |     0.470 |     0.010 |     0.407 |     0.142 |     0.713 |     0.785 | **0.719** |
| yolov8         |     0.352 |     0.018 |     0.335 |     0.117 |     0.654 |     0.532 |     0.386 |

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

- **ARTF_Shorts is poor for both yolo models** (PCK@0.05 of 7.4 and 10.2, mAP of 0.010 and 0.018, and not a
  single image with all keypoints correct). It is the smallest training split in the benchmark, so at 100
  epochs and batch 16 it gets only ~600 optimiser steps, against ~57,000 for AP-10K. The budget is
  specified in epochs, so it scales with dataset size; these cells are likely undertrained rather than
  genuinely hard.
- **`yolo26s` detects nothing in 21% of the ARTF_Tshirts test set**, which is what its detection rate of
  78.8 reports. Those images are counted as failures by PCK, strict success and mAP, but are still skipped
  by the raw pixel distances in `metrics.csv`.
- **Strict success is far below PCK on the many-keypoint datasets.** AP-10K has 17 keypoints and the best
  model gets 68.0 PCK but only 11.4 of its images completely right. That is arithmetic rather than a
  surprise, but it is the number that matters if the whole keypoint configuration has to be correct.
- **mAP is much lower than PCK on AP-10K and CUB-200** because it is the only metric that charges for
  predicting a keypoint that is out of view, and 41% respectively 20% of their keypoint slots are. Both
  model families always emit every keypoint channel, so both pay for it.


## Local Development

### Local installation

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.
