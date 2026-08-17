"""Prepare the GITW (Glasses-in-the-Wild) dataset for this benchmark.

https://zenodo.org/records/17288503 (CC-BY-4.0)
Adriaens, Lips, De Coster, Verleysen, wyffels (Ghent University).

1000 images of transparent and partially filled glasses, annotated with 5 keypoints:
bottom_front, top_left, top_right, top_front and fluid_level.

The published archives already contain a usable coco keypoints file per split, at
`<split>/images/annotations.json`, using the anonymised file names that match the images on disk.
(The `annotations.xml` next to it is a stale CVAT export that still refers to the original,
pre-anonymisation file names, and shares no file name with the shipped images. It is ignored.)

Two things still have to be done here:

    - The published val split becomes our test split, and a new val split is carved out of the
      published train split, so that model selection never touches the reported test set.
    - The annotations carry no bounding boxes, which coco_instances_to_yolo needs, so a box is
      derived from the extent of the keypoints.

The images are already 256x256, so no resizing step is needed.
"""

import json
import os
import random
import shutil
from pathlib import Path

from kp_2d_benchmark import DATASET_DIR
from kp_2d_benchmark.eval.coco_results import (
    CocoImage,
    CocoKeypointAnnotation,
    CocoKeypointCategory,
    CocoKeypointsDataset,
)

GITW_DIR = DATASET_DIR / "gitw"
GITW_RAW_DIR = GITW_DIR / "raw"
GITW_256_DIR = GITW_DIR / "256x256"

ZENODO_TRAIN_URL = "https://zenodo.org/records/17288503/files/glasses_in_the_wild_1000_train.zip?download=1"
ZENODO_VAL_URL = "https://zenodo.org/records/17288503/files/glasses_in_the_wild_1000_val.zip?download=1"

CATEGORY_NAME = "glass"

# fraction of the published train split that is held out as our validation split.
VAL_FRACTION = 0.1
SPLIT_SEED = 2024

# the keypoints only span the glass, so pad the derived box a little.
BBOX_MARGIN = 0.1


def download_raw_dataset(override: bool = False):
    if GITW_RAW_DIR.exists() and not override:
        print(f"Folder {GITW_RAW_DIR} already exists, assuming dataset was already downloaded.")
        return

    GITW_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in (("train", ZENODO_TRAIN_URL), ("val", ZENODO_VAL_URL)):
        zip_path = GITW_DIR / f"gitw_{name}.zip"
        os.system(f"wget '{url}' -O {zip_path}")
        os.system(f"unzip -q {zip_path} -d {GITW_RAW_DIR}")
        os.remove(zip_path)


def bbox_from_keypoints(keypoints, image_width, image_height, margin=BBOX_MARGIN):
    """Axis aligned box around the labeled keypoints, padded and clipped to the image."""
    xs = [keypoints[i] for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]
    ys = [keypoints[i + 1] for i in range(0, len(keypoints), 3) if keypoints[i + 2] > 0]
    assert xs and ys, "cannot derive a bbox for an annotation without labeled keypoints"

    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin
    x0, x1 = max(x0 - pad_x, 0), min(x1 + pad_x, image_width)
    y0, y1 = max(y0 - pad_y, 0), min(y1 + pad_y, image_height)
    return (x0, y0, max(x1 - x0, 1.0), max(y1 - y0, 1.0))


def read_published_split(split_dir):
    """Read a published split, whose coco file lives inside the images folder."""
    image_dir = Path(split_dir) / "images"
    with open(image_dir / "annotations.json") as f:
        return json.load(f), image_dir


def write_split(raw, image_dir, image_ids, target_dir):
    """Write the given images of a published split as a standalone coco dataset with copied images."""
    target_dir = Path(target_dir)
    (target_dir / "images").mkdir(parents=True, exist_ok=True)

    raw_category = raw["categories"][0]
    category = CocoKeypointCategory(
        supercategory=CATEGORY_NAME,
        id=raw_category["id"],
        name=raw_category["name"],
        keypoints=raw_category["keypoints"],
        skeleton=raw_category.get("skeleton"),
    )

    keep = set(image_ids)
    raw_images = {image["id"]: image for image in raw["images"] if image["id"] in keep}

    images = []
    for image in raw_images.values():
        # the published file_name is a bare basename, we store the images in an images/ subfolder
        # like the other datasets do.
        shutil.copyfile(image_dir / image["file_name"], target_dir / "images" / image["file_name"])
        images.append(
            CocoImage(
                id=image["id"],
                file_name=f"images/{image['file_name']}",
                width=image["width"],
                height=image["height"],
            )
        )

    annotations = []
    for annotation in raw["annotations"]:
        if annotation["image_id"] not in keep:
            continue
        image = raw_images[annotation["image_id"]]
        annotations.append(
            CocoKeypointAnnotation(
                id=annotation["id"],
                image_id=annotation["image_id"],
                category_id=annotation["category_id"],
                keypoints=annotation["keypoints"],
                num_keypoints=annotation.get("num_keypoints"),
                bbox=bbox_from_keypoints(annotation["keypoints"], image["width"], image["height"]),
            )
        )

    dataset = CocoKeypointsDataset(categories=[category], images=images, annotations=annotations)
    with open(target_dir / "annotations.json", "w") as f:
        json.dump(dataset.model_dump(exclude_none=True), f)

    print(f"{target_dir.name}: {len(images)} images, {len(annotations)} annotations")


def convert():
    published_train = GITW_RAW_DIR / "glasses_in_the_wild_1000_train"
    published_val = GITW_RAW_DIR / "glasses_in_the_wild_1000_val"
    assert published_train.exists() and published_val.exists(), "raw dataset not found, download it first"

    # the published validation split becomes our test set.
    raw_val, val_image_dir = read_published_split(published_val)
    write_split(raw_val, val_image_dir, [image["id"] for image in raw_val["images"]], GITW_256_DIR / "test")

    # and a fresh validation split is carved out of the published train split.
    raw_train, train_image_dir = read_published_split(published_train)
    image_ids = sorted(image["id"] for image in raw_train["images"])
    random.Random(SPLIT_SEED).shuffle(image_ids)
    num_val = round(len(image_ids) * VAL_FRACTION)

    write_split(raw_train, train_image_dir, image_ids[num_val:], GITW_256_DIR / "train")
    write_split(raw_train, train_image_dir, image_ids[:num_val], GITW_256_DIR / "val")


if __name__ == "__main__":
    download_raw_dataset()
    convert()
