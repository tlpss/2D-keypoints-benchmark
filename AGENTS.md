# AGENTS.md

Guidance for coding agents working in this repository.

## What this is

A benchmark for 2D category-level keypoint detection. It trains a fixed set of models on a fixed set of
datasets and reports keypoint distance metrics. The deliverable is the table in `README.md`, generated from
`metrics.csv`.

Adding a dataset or a model means **retraining and regenerating that table**, not just adding code.

## Environment

Always use the project conda environment. Do not use the base interpreter:

```
/home/tlips/.conda/envs/2D-keypoints-benchmark/bin/python
```

The code uses Python 3.12 syntax (PEP 701 nested-quote f-strings), so a 3.10/3.11 interpreter fails at
*parse* time with a confusing `SyntaxError` in an unrelated file. If an import blows up in
`kp_2d_benchmark/datasets/artf.py`, you are on the wrong interpreter.

`environment.yaml` is incomplete: torch and the `keypoint-detection` submodule are installed by hand and
not declared there. `ultralytics` is pinned, deliberately (see below).

## Layout

- `kp_2d_benchmark/datasets/` — one module per dataset, plus `<name>_to_coco.py` conversion scripts
- `kp_2d_benchmark/eval/` — metrics and the COCO pydantic parsers
- `scripts/pl_keypoint_detector.py` — trains the heatmap models (`MaxVitUnet`, `DinoV2Up`)
- `scripts/yolo.py` — trains ultralytics pose models
- `data/` — **entirely gitignored** (`data/.gitignore` is `**`). Datasets, results and run logs all live here
  and are never committed. Only `metrics.csv` at the repo root and the README tables are tracked.

## Adding a dataset

1. Write `kp_2d_benchmark/datasets/<name>_to_coco.py` that converts the upstream release into COCO
   keypoints format. Reuse the pydantic models in `kp_2d_benchmark/eval/coco_results.py`
   (`CocoKeypointsDataset` etc.) — constructing through them validates the output for free. Reuse
   `resize_coco_dataset` from airo-dataset-tools rather than writing a resizer.
2. Mirror the converted dataset to huggingface under `tlpss/...` and add a `DatasetContainer` subclass whose
   `download()` calls `huggingface_hub.snapshot_download`. Follow `roboflow_garlic.py` as the template.
3. Register the container in `DATASETS` in `kp_2d_benchmark/datasets/__init__.py`.
4. Add tests for the conversion. See `test/test_ap10k_conversion.py` and `test/test_gitw_conversion.py`.
5. Train every model on it and regenerate `metrics.csv` and the README tables.

### Invariants a new dataset must satisfy

- **Exactly one annotation per image.** The distance metrics match predictions to ground truth by
  `image_id` alone and raise on a second annotation. Multi-instance evaluation is deliberately out of
  scope; see the docstring of `kp_2d_benchmark/eval/calculate_keypoint_distance_metrics.py`. AP-10K is
  cropped per instance for exactly this reason.
- **One category, matching `DatasetContainer.category_name`.** `num_keypoints` resolves keypoints by
  matching that name against the `categories` list. Multi-category sources must be collapsed or split.
- **Bounding boxes present.** `coco_instances_to_yolo` unpacks `annotation.bbox` and crashes without one.
  Derive boxes from the keypoint extent if the upstream data has none (see `gitw_to_coco.py`).
- **Images resized to a fixed size**, since the models train at fixed input resolution.

## Results and metrics

Result files are named `data/results/model=<label>,dataset=<DatasetRepr>.json`. That filename **is** the
schema: `calculate_all_metrics.py` parses the model and dataset out of it. Get it wrong and the row is
mislabeled or the script fails to find the dataset.

Regenerate the table with:

```
python kp_2d_benchmark/eval/calculate_all_metrics.py
```

It rewrites `metrics.csv` from every `.json` directly in `data/results/` (subdirectories are skipped, which
is how superseded results are archived).

Five metrics are reported, defined in the "Metrics" section of the README: detection rate, median NME,
PCK@0.05, strict success@0.05 and mAP@0.05. Things worth knowing before touching them:

- **They disagree about undetected images on purpose.** PCK, strict success and mAP charge for them; the
  median NME and the legacy raw-pixel distance columns skip them. Do not "fix" that inconsistency without
  reading the README first.
- **mAP is the submodule's metric**, `keypoint_detection.models.metrics.KeypointAPMetric`, not COCO OKS.
  It is also the only metric that penalises predicting an out-of-view keypoint, which is why it sits far
  below PCK on AP-10K and CUB-200.
- **`AP_ALPHA`, `PCK_ALPHA` and `STRICT_SUCCESS_ALPHA` are benchmark constants.** They are all 0.05 and
  share the same normaliser, `max(bbox_width, bbox_height)`. Changing one changes every number in its
  column, so treat them like the pinned dependencies below.
- **The raw-pixel columns stay in `metrics.csv`** for continuity, and their value must not change when the
  evaluation code is refactored. `test/test_metrics.py` pins the behaviour they depend on; the numbers
  themselves can only be checked against a previous `metrics.csv`, since the result files are gitignored.

## Reproducibility: the two pinned dependencies

Results depend on code outside this repo, and both have already caused mislabeled numbers:

- **`keypoint-detection` submodule.** Keep it pinned. It previously pointed at a commit containing no
  DinoV2 backbone at all while the working tree was several commits ahead, which is how a results column
  ended up labeled `DinoV2Linear` while being produced by a different architecture.
- **`ultralytics`.** Pinned in `environment.yaml`. The version materially changes yolo results in *both*
  directions — going 8.3.58 to 8.4.120 moved AP-10K mean distance 87.6 to 35.6 and CUB 26.0 to 13.5, but
  ARTF tshirts 26.9 to 55.0. Never mix versions within a row of the table; retrain the whole model column.

Training is otherwise deterministic: `seed=2024` for the heatmap models, and re-running an unchanged
configuration reproduces its result file byte-for-byte.

## Gotchas

- **Ultralytics ignores the `runs_dir` setting** and may write to `./runs`. Always take the checkpoint path
  from `model.trainer.best` rather than constructing it. A completed 100-epoch run was once lost to a
  `FileNotFoundError` on the line after training finished.
- **Serialise huggingface uploads.** Concurrent `upload_large_folder` calls get rate-limited with a 429 that
  surfaces as an opaque LFS error. Use `upload_large_folder`, not `huggingface-cli upload`, for many files.
- **pre-commit hooks may run under an older Python than the project.** Nested same-quote f-strings then fail
  with `E999`. Use `f"...{d['k']}..."`, not `f"...{d["k"]}..."`.
- **CI (`.github/workflows/pre-commit.yaml`) runs `pre-commit --all-files`**, so a formatting problem
  anywhere fails the build, not just in changed files.
- The yolo script writes a timestamped `*_data.yaml` temp file into the repo root and only deletes it on the
  success path; interrupted runs leave it, plus a multi-GB copy of the dataset under `data/yolo/`.

## Conventions

- Formatting is enforced by pre-commit: black and isort at line length 119, flake8 at 120, autoflake.
  Run `pre-commit run --files <changed files>` before committing; the first run often reformats and aborts
  the commit, so stage and commit again.
- `pytest test` should pass. Tests cover the dataset conversions, which is where the subtle bugs live.
- Long training runs belong in the background, queued per GPU. There are two GPUs.
