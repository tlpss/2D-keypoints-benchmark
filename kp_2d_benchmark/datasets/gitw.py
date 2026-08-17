"""GITW, Glasses-in-the-Wild: https://zenodo.org/records/17288503 (CC-BY-4.0)
Adriaens, Lips, De Coster, Verleysen, wyffels (Ghent University).

1000 crowdsourced images of transparent and partially filled drinking glasses, with 5 keypoints per
glass: bottom_front, top_left, top_right, top_front and fluid_level. The object is defined mostly by
refraction and highlights rather than texture, and fluid_level is a state dependent keypoint that is
not tied to a fixed part of the object, which makes this different from everything else in the benchmark.

The published validation split is used as our test split, and a new validation split is carved out of
the published train split, see gitw_to_coco.py.
"""

import huggingface_hub

from kp_2d_benchmark import DATASET_DIR
from kp_2d_benchmark.datasets.base import DatasetContainer

GITW_DATASET_256_HF_REPO = "tlpss/gitw-256x256"
GITW_256_DIR = DATASET_DIR / "gitw" / "256x256"


def download_gitw_dataset_256_hf(override: bool = False):
    if not GITW_256_DIR.exists() and not override:
        GITW_256_DIR.mkdir(parents=True)
        huggingface_hub.snapshot_download(GITW_DATASET_256_HF_REPO, repo_type="dataset", local_dir=GITW_256_DIR)


class GITW_256(DatasetContainer):
    json_train_path = GITW_256_DIR / "train" / "annotations.json"
    json_val_path = GITW_256_DIR / "val" / "annotations.json"
    json_test_path = GITW_256_DIR / "test" / "annotations.json"
    category_name = "glass"

    def download(override: bool = False):
        download_gitw_dataset_256_hf(override=override)


if __name__ == "__main__":
    download_gitw_dataset_256_hf()
