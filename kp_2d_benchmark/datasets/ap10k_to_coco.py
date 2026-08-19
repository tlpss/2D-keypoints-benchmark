"""Convert AP-10K into the single-category, instance-cropped COCO format used by this benchmark.

AP-10K: https://github.com/AlexTheBad/AP-10K (CC-BY-4.0)
Yu et al., "AP-10K: A Benchmark for Animal Pose Estimation in the Wild", NeurIPS 2021 Datasets & Benchmarks.

The upstream annotations declare 54 categories (one per species) that all share the same 17-keypoint
quadruped skeleton, and they contain ~1.3 annotated animals per image. Neither fits this benchmark:
DatasetContainer resolves keypoints through a single category name, and the distance metrics assume
at most one annotation per image.

So this script
    - collapses the 54 species into a single "animal" category, and
    - crops every instance into its own image (square, bbox + 25% margin),
which makes the result single-category and single-instance, like the other datasets here.

Usage: download the raw dataset (see download_raw_dataset), then run this file. It writes the cropped
dataset, resizes it to 512x512 with airo-dataset-tools and leaves the result in

    data/datasets/ap10k/512x512/{train,val,test}/

Note that cropping to the ground truth boxes turns AP-10K into a top-down task that assumes a perfect
detector. That is deliberate: it keeps the dataset object-centric like the other five, and it avoids
the instance matching that the distance metrics do not implement.
"""

import json
import os
from pathlib import Path

from airo_dataset_tools.coco_tools.transform_dataset import resize_coco_dataset
from PIL import Image

from kp_2d_benchmark import DATASET_DIR
from kp_2d_benchmark.eval.coco_results import (
    CocoImage,
    CocoKeypointAnnotation,
    CocoKeypointCategory,
    CocoKeypointsDataset,
)

AP10K_DIR = DATASET_DIR / "ap10k"
AP10K_RAW_DIR = AP10K_DIR / "raw"
AP10K_CROPPED_DIR = AP10K_DIR / "cropped"
AP10K_512_DIR = AP10K_DIR / "512x512"

# the whole dataset (images + annotations) is only distributed through google drive.
AP10K_GDRIVE_ID = "1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad"

# AP-10K ships three splits for variance estimation, we use the first one.
SPLIT = "split1"

CATEGORY_NAME = "animal"
CATEGORY_ID = 1

# fraction of the bbox size that is added as context around the instance before cropping.
CROP_MARGIN = 0.25


def download_raw_dataset():
    """Download and extract the raw AP-10K dataset. Requires `pip install gdown`."""
    if AP10K_RAW_DIR.exists():
        print(f"Folder {AP10K_RAW_DIR} already exists, assuming dataset was already downloaded.")
        return

    AP10K_RAW_DIR.mkdir(parents=True)
    zip_path = AP10K_DIR / "ap10k.zip"
    os.system(f"gdown {AP10K_GDRIVE_ID} -O {zip_path}")
    os.system(f"unzip {zip_path} -d {AP10K_RAW_DIR}")


def get_shared_keypoint_definition(categories):
    """AP-10K defines one category per species, all with the same keypoints. Return that shared definition."""
    keypoints = categories[0]["keypoints"]
    skeleton = categories[0].get("skeleton")
    for category in categories:
        assert (
            category["keypoints"] == keypoints
        ), f"category {category['name']} does not share the keypoints of {categories[0]['name']}"
    return keypoints, skeleton


def compute_square_crop_box(bbox, image_width, image_height, margin=CROP_MARGIN):
    """Square crop box around a coco bbox, as (x0, y0, x1, y1) integers.

    The box is expanded by `margin` and made square, and is then shifted to lie within the image where
    that is possible. It is deliberately not clipped: when the image is smaller than the box, the crop
    is zero-padded instead, so that every crop is square and can be resized without distortion.
    """
    x, y, width, height = bbox
    size = (1 + margin) * max(width, height)

    x0 = x + width / 2 - size / 2
    y0 = y + height / 2 - size / 2

    # prefer real pixels over padding: shift the box inwards if it hangs over an edge and still fits.
    if size <= image_width:
        x0 = min(max(x0, 0), image_width - size)
    if size <= image_height:
        y0 = min(max(y0, 0), image_height - size)

    x0, y0, size = round(x0), round(y0), round(size)
    return x0, y0, x0 + size, y0 + size


def crop_keypoints(keypoints, x0, y0, crop_width, crop_height):
    """Translate coco keypoints into the crop frame, marking those that fall outside it as not labeled."""
    cropped = []
    for i in range(0, len(keypoints), 3):
        x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        x, y = x - x0, y - y0
        if visibility == 0 or not (0 <= x < crop_width and 0 <= y < crop_height):
            cropped.extend([0.0, 0.0, 0])
        else:
            cropped.extend([x, y, visibility])
    return cropped


def crop_bbox(bbox, x0, y0, crop_width, crop_height):
    x, y, width, height = bbox
    x, y = x - x0, y - y0
    # clip to the crop, the bbox can stick out of it when the crop was zero-padded.
    x1, y1 = min(x + width, crop_width), min(y + height, crop_height)
    x, y = max(x, 0), max(y, 0)
    return (x, y, max(x1 - x, 1.0), max(y1 - y, 1.0))


def convert_split(raw_json_path, raw_image_dir, target_dir):
    """Crop every instance of a raw AP-10K split into its own image and write a coco keypoints dataset."""
    with open(raw_json_path) as f:
        raw = json.load(f)

    keypoint_names, skeleton = get_shared_keypoint_definition(raw["categories"])
    category = CocoKeypointCategory(
        supercategory=CATEGORY_NAME,
        id=CATEGORY_ID,
        name=CATEGORY_NAME,
        keypoints=keypoint_names,
        skeleton=skeleton,
    )

    raw_images = {image["id"]: image for image in raw["images"]}

    target_dir = Path(target_dir)
    (target_dir / "images").mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    skipped = 0
    for raw_annotation in raw["annotations"]:
        raw_image = raw_images[raw_annotation["image_id"]]

        x0, y0, x1, y1 = compute_square_crop_box(raw_annotation["bbox"], raw_image["width"], raw_image["height"])
        crop_width, crop_height = x1 - x0, y1 - y0

        keypoints = crop_keypoints(raw_annotation["keypoints"], x0, y0, crop_width, crop_height)
        num_keypoints = sum(1 for visibility in keypoints[2::3] if visibility > 0)
        if num_keypoints == 0:
            # nothing left to learn from or to evaluate on.
            skipped += 1
            continue

        # one image per instance, the annotation id is unique so we can use it for both.
        file_name = f"images/{Path(raw_image['file_name']).stem}_{raw_annotation['id']}.jpg"
        # PIL zero-pads the parts of the box that fall outside the image, which keeps the crop square.
        with Image.open(raw_image_dir / raw_image["file_name"]) as image:
            image.convert("RGB").crop((x0, y0, x1, y1)).save(target_dir / file_name, quality=95)

        images.append(CocoImage(id=raw_annotation["id"], file_name=file_name, width=crop_width, height=crop_height))
        annotations.append(
            CocoKeypointAnnotation(
                id=raw_annotation["id"],
                image_id=raw_annotation["id"],
                category_id=CATEGORY_ID,
                keypoints=keypoints,
                num_keypoints=num_keypoints,
                bbox=crop_bbox(raw_annotation["bbox"], x0, y0, crop_width, crop_height),
            )
        )

    dataset = CocoKeypointsDataset(categories=[category], images=images, annotations=annotations)
    with open(target_dir / "annotations.json", "w") as f:
        json.dump(dataset.model_dump(exclude_none=True), f)

    print(f"{raw_json_path.name}: {len(annotations)} instances written, {skipped} skipped (no visible keypoints)")


def normalize_num_keypoints(json_path):
    """Recompute num_keypoints and drop empty annotations, after a resize moved keypoints out of the image.

    resize_coco_dataset sets keypoints that end up outside the resized image to (0,0,0) but does not update
    num_keypoints, which would make the result fail validation when it is parsed back as a CocoKeypointsDataset.
    """
    with open(json_path) as f:
        data = json.load(f)

    annotations = []
    for annotation in data["annotations"]:
        annotation["num_keypoints"] = sum(1 for visibility in annotation["keypoints"][2::3] if visibility > 0)
        if annotation["num_keypoints"] > 0:
            annotations.append(annotation)

    dropped = len(data["annotations"]) - len(annotations)
    annotated_image_ids = {annotation["image_id"] for annotation in annotations}
    data["annotations"] = annotations
    data["images"] = [image for image in data["images"] if image["id"] in annotated_image_ids]

    with open(json_path, "w") as f:
        json.dump(data, f)

    if dropped:
        print(f"{json_path}: dropped {dropped} annotations that lost all keypoints during the resize")


def find_raw_dirs():
    """Locate the annotations and image folders in the extracted archive, whose top level folder varies."""
    matches = list(AP10K_RAW_DIR.glob(f"**/ap10k-train-{SPLIT}.json"))
    assert matches, f"could not find ap10k-train-{SPLIT}.json under {AP10K_RAW_DIR}, is the dataset downloaded?"
    raw_annotations_dir = matches[0].parent
    raw_image_dir = raw_annotations_dir.parent / "data"
    assert raw_image_dir.exists(), f"expected the images next to the annotations, at {raw_image_dir}"
    return raw_annotations_dir, raw_image_dir


def convert(width=512, height=512):
    raw_annotations_dir, raw_image_dir = find_raw_dirs()

    for split in ("train", "val", "test"):
        cropped_dir = AP10K_CROPPED_DIR / split
        convert_split(raw_annotations_dir / f"ap10k-{split}-{SPLIT}.json", raw_image_dir, cropped_dir)

        resized_dir = AP10K_512_DIR / split
        resize_coco_dataset(str(cropped_dir / "annotations.json"), width, height, target_dataset_dir=str(resized_dir))
        normalize_num_keypoints(resized_dir / "annotations.json")


if __name__ == "__main__":
    download_raw_dataset()
    convert()
