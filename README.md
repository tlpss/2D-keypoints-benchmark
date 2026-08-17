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

**Distances are in raw pixels.** The reported distances are not normalised by object or image size, so they
are only directly comparable between datasets of equal resolution.

**Images without a prediction are skipped.** If a model detects nothing in an image, that image does not
contribute to the distance metrics rather than being penalised, so the metrics are not detection-aware.


## Performance numbers

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
