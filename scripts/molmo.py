"""Zero-shot keypoint prediction with Molmo, as a VLM baseline for the benchmark.

Molmo is trained to *point*: given "point to the X" it answers with a `<point x=".." y=".."/>` tag
whose coordinates are percentages of the image size. This script asks it for one keypoint at a time,
converts the answer to pixels and writes a result file in the benchmark's usual format, so that
`calculate_all_metrics.py` picks it up like any other model.

Run it from a *separate* environment, not the benchmark one. transformers and its dependencies can
pull in a different torch, and the pinned ultralytics and keypoint-detection versions are what make
the existing rows of the table reproducible. The result file is the only interface that matters:

    conda create -n molmo python=3.12
    conda activate molmo
    pip install torch transformers accelerate einops torchvision pillow
    PYTHONPATH=/path/to/2D-keypoints-benchmark python scripts/molmo.py --dataset RoboflowGarlic256Dataset

Two things to know about the numbers this produces:

- **Molmo has no notion of per-keypoint confidence.** Every keypoint it successfully points at is
  written with a score of 1.0, so the mAP column measures precision and recall but not confidence
  ranking, and the sparsification diagnostics are meaningless for this row. This is a property of
  the model, not of the harness.
- **A keypoint the model refuses to point at has no representation in the result format**, which is
  per-image rather than per-keypoint. The policy here: if *every* keypoint fails, the image is left
  out of the results entirely, which the metrics count as a missed detection. If only some fail, the
  missing ones fall back to the image centre with a score of 0.0, so they rank last and count as
  localisation failures. The summary printed at the end reports how often each happened.
"""

import argparse
import datetime
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from kp_2d_benchmark import DATA_DIR
from kp_2d_benchmark.datasets import DATASETS
from kp_2d_benchmark.datasets.base import DatasetContainer
from kp_2d_benchmark.eval.coco_results import COCOKeypointResult, COCOKeypointResults

DEFAULT_MODEL = "allenai/Molmo-7B-D-0924"
DEFAULT_DATASET = "RoboflowGarlic256Dataset"

# Molmo answers a pointing prompt with either a single <point> or a <points> tag holding several
# candidates. Coordinates are percentages of the image dimensions, not pixels.
POINT_PATTERN = re.compile(r'x(?:\d+)?\s*=\s*"([\d.]+)"\s*y(?:\d+)?\s*=\s*"([\d.]+)"')

# A full noun phrase per keypoint, since the bare names are only meaningful to someone who knows the
# dataset. Anything not listed falls back to "the <name> of the <category>", which reads fine for the
# datasets whose names are already descriptive ("the left front paw of the animal").
KEYPOINT_PHRASINGS = {
    "RoboflowGarlic256Dataset": {
        "head": "the root end of the garlic bulb, where the roots come out",
        "tail": "the stem end of the garlic bulb, where the dried stalk sticks out",
    },
    "GITW_256": {
        "bottom_front": "the bottom edge of the glass, nearest the viewer",
        "top_left": "the left edge of the rim of the glass",
        "top_right": "the right edge of the rim of the glass",
        "top_front": "the front edge of the rim of the glass, nearest the viewer",
        "fluid_level": "the surface of the liquid in the glass",
    },
}


def describe_keypoint(dataset_name: str, keypoint_name: str, category_name: str) -> str:
    phrasing = KEYPOINT_PHRASINGS.get(dataset_name, {}).get(keypoint_name)
    if phrasing is not None:
        return phrasing
    return f"the {keypoint_name.replace('_', ' ')} of the {category_name}"


def build_prompt(keypoint_description: str) -> str:
    return (
        f"Point to {keypoint_description} in this image. "
        "If it is not visible, say 'not visible' instead of pointing."
    )


def parse_point(answer: str, width: int, height: int) -> Optional[Tuple[float, float]]:
    """The first point of a Molmo answer, in pixels, or None if it did not point at anything.

    Molmo's coordinates are percentages of the image size, so they have to be scaled by the actual
    image dimensions. A <points> tag holds several candidates; the first is used.
    """
    match = POINT_PATTERN.search(answer)
    if match is None:
        return None
    x_percent, y_percent = float(match.group(1)), float(match.group(2))
    return x_percent / 100.0 * width, y_percent / 100.0 * height


class MolmoPointer:
    """Thin wrapper around the Molmo checkpoint, loaded once and reused for every prompt."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.device = device
        # bfloat16 explicitly: the released checkpoints are float32 and torch_dtype="auto" would try
        # to put ~32GB on the card, which does not fit on a 24GB GPU and looks like the model being
        # too large rather than a dtype default.
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device
        )

    def point(self, image: Image.Image, prompt: str, max_new_tokens: int = 64) -> str:
        from transformers import GenerationConfig

        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def predict_dataset(  # noqa: C901
    dataset: DatasetContainer, pointer: MolmoPointer, limit: Optional[int] = None, verbose: bool = False
) -> Tuple[List[COCOKeypointResult], dict]:
    """Point at every keypoint of every test image, and collect the results."""
    with open(dataset.json_test_path) as f:
        data = json.load(f)

    category = next((c for c in data["categories"] if c["name"] == dataset.category_name), None)
    if category is None:
        raise ValueError(f"Category {dataset.category_name} not found in the dataset")
    keypoint_names = category["keypoints"]

    images = data["images"][:limit] if limit else data["images"]
    image_dir = Path(dataset.json_test_path).parent

    results: List[COCOKeypointResult] = []
    stats = {"images": len(images), "skipped_images": 0, "keypoints": 0, "missed_keypoints": 0}

    for index, image_entry in enumerate(images):
        image = Image.open(image_dir / image_entry["file_name"]).convert("RGB")
        width, height = image.size

        keypoints: List[float] = []
        scores: List[float] = []
        for keypoint_name in keypoint_names:
            prompt = build_prompt(describe_keypoint(repr(dataset), keypoint_name, dataset.category_name))
            answer = pointer.point(image, prompt)
            point = parse_point(answer, width, height)
            stats["keypoints"] += 1

            if point is None:
                # no point in the answer: fall back to the image centre so the keypoint still has a
                # slot, and score it 0.0 so it ranks last in the AP computation.
                stats["missed_keypoints"] += 1
                keypoints.extend([width / 2, height / 2, 2])
                scores.append(0.0)
                if verbose:
                    print(f"    {keypoint_name}: no point in answer {answer.strip()!r}")
            else:
                keypoints.extend([point[0], point[1], 2])
                scores.append(1.0)
                if verbose:
                    print(f"    {keypoint_name}: ({point[0]:.1f}, {point[1]:.1f})")

        if all(score == 0.0 for score in scores):
            # the model pointed at nothing at all in this image, which is a missed detection rather
            # than a set of bad keypoints.
            stats["skipped_images"] += 1
            print(f"[{index + 1}/{len(images)}] image {image_entry['id']}: no keypoints found, skipping")
            continue

        results.append(
            COCOKeypointResult(
                image_id=image_entry["id"],
                category_id=category["id"],
                keypoints=keypoints,
                score=sum(scores) / len(scores),
                per_keypoint_scores=scores,
            )
        )
        print(
            f"[{index + 1}/{len(images)}] image {image_entry['id']}: {sum(s > 0 for s in scores)}/{len(scores)} found"
        )

    return results, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="DatasetContainer repr, e.g. GITW_256")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="huggingface checkpoint")
    parser.add_argument("--model-label", default=None, help="label used in the result file name")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:1, cpu, ...")
    parser.add_argument("--limit", type=int, default=None, help="only run on the first N images")
    parser.add_argument("--verbose", action="store_true", help="print every point and every failed answer")
    args = parser.parse_args()

    dataset = next((d for d in DATASETS if repr(d) == args.dataset), None)
    if dataset is None:
        raise SystemExit(f"Dataset {args.dataset} not found. Available: {[repr(d) for d in DATASETS]}")

    model_label = args.model_label or args.model.split("/")[-1]
    # a --limit run covers part of the test set, so its detection rate is meaningless as a table row.
    # park it in a subdirectory, which calculate_all_metrics.py skips, rather than let a smoke test
    # quietly become a published number.
    results_dir = DATA_DIR / "results" / "dry-run" if args.limit else DATA_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # the file name is the schema that calculate_all_metrics.py parses, so it has to be exactly this
    results_path = results_dir / f"model={model_label},dataset={args.dataset}.json"

    print(f"loading {args.model} onto {args.device}")
    pointer = MolmoPointer(args.model, args.device)

    start = datetime.datetime.now()
    results, stats = predict_dataset(dataset, pointer, limit=args.limit, verbose=args.verbose)
    elapsed = datetime.datetime.now() - start

    if not results:
        raise SystemExit("the model did not point at anything in any image; not writing a result file")

    with open(results_path, "w") as f:
        f.write(COCOKeypointResults(root=results).model_dump_json())

    print(f"\n{stats['images']} images in {elapsed} ({elapsed / stats['images']} per image)")
    print(f"detected {len(results)}/{stats['images']} images ({stats['skipped_images']} with no keypoint at all)")
    print(f"pointed at {stats['keypoints'] - stats['missed_keypoints']}/{stats['keypoints']} keypoints")
    print(f"\nwrote {results_path}")
    print("regenerate the table with: python kp_2d_benchmark/eval/calculate_all_metrics.py")


if __name__ == "__main__":
    main()
