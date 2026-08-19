"""One-shot keypoint detection with frozen ViT features, as the data-efficiency baseline of the benchmark.

The method comes from the `few-shot-keypoints` submodule and is training-free. A frozen backbone turns an
image into a dense per-pixel feature map; one reference feature vector is sliced out per keypoint channel
from a single annotated support image; prediction on a test image is the argmax of the cosine similarity
between its feature map and that reference vector. So the whole "training" budget is **one annotated point
per keypoint**, against the 83-9122 training images the other rows of the table use.

Run it from the submodule's environment, not the benchmark one. `few-shot-keypoints` needs
transformers>=5 and torch>=2.7, while the benchmark env has no transformers at all and a pinned
ultralytics/torch that keep the existing rows reproducible. The result file is the only interface:

    PYTHONPATH=.:few-shot-keypoints few-shot-keypoints/.venv/bin/python \
        scripts/few_shot_matching.py --dataset RoboflowGarlic256Dataset --featurizer dinov3-s

PYTHONPATH must put the submodule first, so that the pinned source runs rather than whatever the venv has
installed as an editable package.

**What actually pins these numbers.** The submodule commit pins the method; the environment is pinned only
by writing the versions down, because the `uv.lock` committed at that commit is *stale* relative to its own
pyproject.toml -- it holds transformers 4.56.0 while the manifest requires >=5.0.0, so `uv sync`
re-resolves rather than replaying the lock, and resolves to whatever is newest on the day it runs. The
venv these results were produced with (`few-shot-keypoints/.venv`, built with `uv sync`) holds:

    torch 2.7.1+cu126, torchvision 0.22.1, transformers 5.15.1, timm 1.0.28, numpy 1.26.4,
    airo-dataset-tools 2025.7.1

A third hazard, specific to RADIO: `nvidia/C-RADIOv2-B` ships its modelling code through
trust_remote_code and it is fetched unpinned, so a change upstream lands in a rerun without warning.
Upstreaming a refreshed `uv.lock` and pinning the RADIO revision would close both gaps.

**Protocol.** Matching happens on an object crop: the annotation's bbox plus a 10% margin, padded to a
square and resized to 512x512, with the match mapped back to the original image frame. The alternative,
matching over the whole image, gives a 16x16 patch grid on a 256x256 dataset, which is too coarse to
localise anything. Two consequences to keep in mind when reading the numbers:

- **It uses the ground-truth bbox at test time**, so it assumes a perfect detector. AP-10K already bakes
  that assumption in (it is cropped per instance), but the other six datasets do not, so this row is not
  comparable to the trained models on detection, only on localisation.
- **Segmentation-mask matching is not used**, because four of the seven datasets have no masks.

**One shot means high variance.** Which train image supplies the reference point is chosen by `seed`, and
a single unlucky reference can move a whole row. So every configuration is run at five seeds, written to
`data/results/few-shot-seeds/`, and `calculate_all_metrics.py` reports the mean over them plus the spread
(see `kp_2d_benchmark/eval/aggregate_seed_metrics.py`). Never promote a single-seed file into
`data/results/` directly.

Two properties of these rows that are the method's, not the harness's:

- **Detection rate is 100% by construction.** The argmax always returns a pixel, so every test image gets
  a full set of keypoints and no image is ever skipped. That column carries no information here, unlike
  for the yolo and Molmo rows where it is the interesting failure mode.
- **The confidence is a raw cosine similarity**, which sits in a narrow band around 0.9 and barely
  separates a good match from a bad one. mAP is still computed, but its ranking component is weak.
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import List, Tuple

from kp_2d_benchmark import DATA_DIR
from kp_2d_benchmark.datasets import DATASETS
from kp_2d_benchmark.datasets.base import DatasetContainer
from kp_2d_benchmark.eval.coco_results import COCOKeypointResult, COCOKeypointResults

DEFAULT_FEATURIZER = "dinov3-s"
DEFAULT_SEEDS = [2025, 2026, 2027, 2028, 2029]
DEFAULT_CROP_SIZE = (512, 512)
DEFAULT_MARGIN_SCALE = 0.1

# the subdirectory the per-seed files go into. `calculate_all_metrics.py` skips subdirectories, so a seed
# file can never become a table row on its own; the aggregator reads them back out of here.
SEED_RESULTS_SUBDIR = "few-shot-seeds"


def filtered_coco_json(dataset: DatasetContainer, split: str) -> Path:
    """A copy of one split of `dataset` that `TorchCOCOKeypointsDataset` will accept, cached under `data/`.

    Two things have to change. The three aRTF datasets declare all four garment categories in every split
    file even though the annotations of a given file are all one category, and the few-shot loader asserts
    a single category and reads the keypoint names off `categories[0]` -- so it would crash on shorts and
    tshirts and silently detect towel's four corners on the others. Only the category whose name matches
    `dataset.category_name` is kept, with its `id` untouched so the emitted `category_id` still matches
    ground truth.

    And image paths are resolved relative to the directory of the json, which this copy does not share, so
    they are rewritten to absolute paths. `Path.__truediv__` returns the right operand when it is
    absolute, which is what makes that work.

    `segmentation` is stripped. Crop matching never reads the mask -- neither the reference sampling nor
    the inference loop of `dataset_object_crop_matching` touches it -- but the loader decodes one per
    image at init anyway, and RoboFlow Garlic ships an empty `segmentation: []` that pycocotools raises an
    IndexError on. Dropping it removes both the crash and the wasted decode.

    Second annotations of an image are dropped, keeping the first in file order. The benchmark documents
    one annotation per image as a dataset invariant, and every *test* split satisfies it, which is why the
    metrics have never had to care -- but the RoboFlow Garlic *train* split does not: 2 of its 697 images
    carry two garlic bulbs, for 699 annotations.

    And `num_keypoints` is recomputed from the visibility flags rather than trusted. It is redundant
    metadata, and CUB-200-2011 train annotation 5007 is the one place in the benchmark where it disagrees
    with the flags (it claims 8 labeled keypoints against 7 actually flagged), which the stricter COCO
    parser this loader uses rejects outright.

    Both of those only touch the *train* split, so they change the pool of candidate reference points and
    never a reported number.
    """
    json_path = Path(getattr(dataset, f"json_{split}_path"))
    with open(json_path) as f:
        data = json.load(f)

    categories = [category for category in data["categories"] if category["name"] == dataset.category_name]
    if not categories:
        raise ValueError(f"Category {dataset.category_name} not found in {json_path}")
    data["categories"] = categories

    image_dir = json_path.parent
    for image in data["images"]:
        image["file_name"] = str(image_dir / image["file_name"])

    annotations, seen_image_ids = [], set()
    for annotation in data["annotations"]:
        if annotation["image_id"] in seen_image_ids:
            continue
        seen_image_ids.add(annotation["image_id"])
        annotation.pop("segmentation", None)
        keypoints = annotation["keypoints"]
        annotation["num_keypoints"] = sum(1 for i in range(len(keypoints) // 3) if keypoints[3 * i + 2] > 0)
        annotations.append(annotation)
    if len(annotations) != len(data["annotations"]):
        dropped = len(data["annotations"]) - len(annotations)
        print(f"{repr(dataset)} {split}: dropped {dropped} second annotation(s) of an already annotated image")
    data["annotations"] = annotations

    target_path = DATA_DIR / "few-shot-coco" / repr(dataset) / f"{split}.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as f:
        json.dump(data, f)
    return target_path


def to_benchmark_results(few_shot_results) -> COCOKeypointResults:
    """Rename the few-shot repo's result fields onto the benchmark's result model.

    The two formats agree on everything that matters -- flattened [u,v,visibility] triplets in pixels, in
    the category's keypoint order, one entry per image -- so this is a field rename rather than a
    conversion. `id` and `bbox` are dropped because the benchmark model has no place for them.
    """
    results = [
        COCOKeypointResult(
            image_id=annotation.image_id,
            category_id=annotation.category_id,
            keypoints=annotation.keypoints,
            score=annotation.score,
            per_keypoint_scores=annotation.keypoint_scores,
        )
        for annotation in few_shot_results.root
    ]
    return COCOKeypointResults(root=results)


def run_seed(
    train_dataset,
    test_dataset,
    featurizer,
    seed: int,
    crop_target_size: Tuple[int, int],
    margin_scale: float,
    device: str,
) -> COCOKeypointResults:
    """Sample a support set for `seed`, then match every test image against it."""
    from few_shot_keypoints.dataset_object_crop_matching import (
        populate_matcher_w_random_references,
        run_coco_dataset_inference,
    )
    from few_shot_keypoints.matcher import KeypointFeatureMatcher

    reference_vectors = populate_matcher_w_random_references(
        train_dataset,
        featurizer,
        crop_target_size=crop_target_size,
        margin_scale=margin_scale,
        seed=seed,
    )
    matcher = KeypointFeatureMatcher(reference_vectors, device=device)
    few_shot_results = run_coco_dataset_inference(
        test_dataset,
        matcher,
        featurizer,
        crop_target_size=crop_target_size,
        margin_scale=margin_scale,
    )
    return to_benchmark_results(few_shot_results)


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="DatasetContainer repr, e.g. GITW_256")
    parser.add_argument("--featurizer", default=DEFAULT_FEATURIZER, help="few-shot-keypoints registry name")
    parser.add_argument("--model-label", default=None, help="label used in the result file name")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="one support set per seed")
    parser.add_argument("--crop-target-size", type=int, nargs=2, default=list(DEFAULT_CROP_SIZE), metavar=("H", "W"))
    parser.add_argument("--margin-scale", type=float, default=DEFAULT_MARGIN_SCALE, help="bbox margin")
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cuda:1, cpu, ...")
    parser.add_argument("--limit", type=int, default=None, help="only run on the first N test images")
    args = parser.parse_args()

    dataset = next((d for d in DATASETS if repr(d) == args.dataset), None)
    if dataset is None:
        raise SystemExit(f"Dataset {args.dataset} not found. Available: {[repr(d) for d in DATASETS]}")

    # importing the package is what populates the registry; without it `create` finds nothing.
    import few_shot_keypoints.featurizers  # noqa: F401
    from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
    from few_shot_keypoints.featurizers.registry import FeaturizerRegistry

    model_label = args.model_label or f"fsk-{args.featurizer}"
    if "," in model_label or "=" in model_label:
        raise SystemExit("the result file name is parsed on ',' and '=', so the label may not contain them")

    # a --limit run covers part of the test set, so its numbers are not a table row. park it in the
    # dry-run subdirectory, which the metrics scripts skip, rather than let a smoke test become a
    # published number.
    results_dir = DATA_DIR / "results" / ("dry-run" if args.limit else SEED_RESULTS_SUBDIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    crop_target_size = tuple(args.crop_target_size)

    train_dataset = TorchCOCOKeypointsDataset(str(filtered_coco_json(dataset, "train")), transform=None)
    test_dataset = TorchCOCOKeypointsDataset(str(filtered_coco_json(dataset, "test")), transform=None)
    if args.limit:
        test_dataset.dataset = test_dataset.dataset[: args.limit]

    print(f"loading {args.featurizer} onto {args.device}")
    featurizer = FeaturizerRegistry.create(args.featurizer, device=args.device)

    written: List[Path] = []
    for seed in args.seeds:
        results_path = results_dir / f"model={model_label},dataset={args.dataset},seed={seed}.json"
        if results_path.exists():
            print(f"skipping seed {seed}, {results_path} already exists")
            continue

        print(f"\n=== {model_label} on {args.dataset}, seed {seed} ({len(test_dataset)} test images) ===")
        start = datetime.datetime.now()
        results = run_seed(
            train_dataset,
            test_dataset,
            featurizer,
            seed,
            crop_target_size,
            args.margin_scale,
            args.device,
        )
        elapsed = datetime.datetime.now() - start

        with open(results_path, "w") as f:
            f.write(results.model_dump_json())
        written.append(results_path)
        print(f"{len(results.root)} predictions in {elapsed} ({elapsed / max(len(results.root), 1)} per image)")
        print(f"wrote {results_path}")

    if written and not args.limit:
        print("\nregenerate the table with: python kp_2d_benchmark/eval/calculate_all_metrics.py")


if __name__ == "__main__":
    main()
