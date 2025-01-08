"""https://universe.roboflow.com/gesture-recognition-dsn2n/garlic_keypoint"""

import huggingface_hub

from kp_2d_benchmark import DATASET_DIR
from kp_2d_benchmark.datasets.base import DatasetContainer

ROBOFLOW_GARLIC_DATASET_256_HF_REPO = "tlpss/roboflow-garlic-256x256"
ROBOFLOW_GARLIC_256_DIR = DATASET_DIR / "roboflow-garlic" / "256"

ROBOFLOW_GARLIC_DATASET_512_HF_REPO = "tlpss/roboflow-garlic-512x512"
ROBOFLOW_GARLIC_512_DIR = DATASET_DIR / "roboflow-garlic" / "512"


def download_roboflow_garlic_dataset_256_hf(override: bool = False):
    if not ROBOFLOW_GARLIC_256_DIR.exists() and not override:
        ROBOFLOW_GARLIC_256_DIR.mkdir(parents=True)
        huggingface_hub.snapshot_download(
            ROBOFLOW_GARLIC_DATASET_256_HF_REPO, repo_type="dataset", local_dir=ROBOFLOW_GARLIC_256_DIR
        )


def download_roboflow_garlic_dataset_512_hf(override: bool = False):
    if not ROBOFLOW_GARLIC_512_DIR.exists() and not override:
        ROBOFLOW_GARLIC_512_DIR.mkdir(parents=True)
        huggingface_hub.snapshot_download(
            ROBOFLOW_GARLIC_DATASET_512_HF_REPO, repo_type="dataset", local_dir=ROBOFLOW_GARLIC_512_DIR
        )


class RoboflowGarlic256Dataset(DatasetContainer):
    json_train_path = ROBOFLOW_GARLIC_256_DIR / "train" / "annotations.json"
    json_val_path = ROBOFLOW_GARLIC_256_DIR / "val" / "annotations.json"
    json_test_path = ROBOFLOW_GARLIC_256_DIR / "test" / "annotations.json"
    category_name = "garlic"

    def download(override: bool = False):
        download_roboflow_garlic_dataset_256_hf(override=override)


class RoboflowGarlic512Dataset(DatasetContainer):
    json_train_path = ROBOFLOW_GARLIC_512_DIR / "train" / "annotations.json"
    json_val_path = ROBOFLOW_GARLIC_512_DIR / "val" / "annotations.json"
    json_test_path = ROBOFLOW_GARLIC_512_DIR / "test" / "annotations.json"

    def download(override: bool = False):
        download_roboflow_garlic_dataset_512_hf(override=override)


if __name__ == "__main__":
    download_roboflow_garlic_dataset_256_hf()
