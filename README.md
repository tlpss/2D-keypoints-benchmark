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
| pkd-DinoV2Up   |        35.4 |                  17.9 |                   9   |                   20.3 |              10   |        7.1 |                        5   |
| pkd-MaxVitUnet |        39.8 |                  40.2 |                  16.1 |                   19   |              11.8 |        5.2 |                        7   |
| yolov8         |        87.6 |                  39   |                  24.5 |                   26.9 |              26   |       16.3 |                       24.2 |

**median keypoint distance**

| model          |   AP10K_512 |   ARTF_Shorts_Dataset |   ARTF_Towels_Dataset |   ARTF_Tshirts_Dataset |   CUB200_2011_512 |   GITW_256 |   RoboflowGarlic256Dataset |
|:---------------|------------:|----------------------:|----------------------:|-----------------------:|------------------:|-----------:|---------------------------:|
| pkd-DinoV2Up   |        12.4 |                   3.9 |                   1.9 |                    3.4 |               6.5 |        2.9 |                        3   |
| pkd-MaxVitUnet |        13.9 |                   6.5 |                   1.2 |                    2.8 |               6.9 |        2.2 |                        3.5 |
| yolov8         |        23.3 |                  20.6 |                   9.1 |                   12.8 |               9.8 |        6.2 |                       10.2 |

Numbers are produced by `kp_2d_benchmark/eval/calculate_all_metrics.py` from the result files in
`data/results/`, and are reproducible: the training uses a fixed seed, and re-running a model on an
unchanged dataset reproduces its result file.

The `keypoint-detection` submodule is pinned, because the backbone that produced an earlier version of
these numbers was renamed and changed upstream while the parent repo pointed at a commit that did not
contain it at all. Keep it pinned when updating results.


## Local Development

### Local installation

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.
