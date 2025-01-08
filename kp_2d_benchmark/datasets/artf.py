"""
artf dataset:

"""

import os

import huggingface_hub

from kp_2d_benchmark import DATASET_DIR
from kp_2d_benchmark.datasets.base import DatasetContainer

ARTF_RESIZED_SPLITS_ZENODO_URL = (
    "https://zenodo.org/records/10875542/files/aRTFClothes-resized-paper-splits.zip?download=1"
)
ARTF_RESIZED_COMBINED_URL = (
    "https://zenodo.org/records/14501499/files/artf-all-categories-resized_512x256.zip?download=1"
)
ARTF_RESIZED_CATEGORIES_HF_URL = "tlpss/artf-categories-512x256"


ARTF_COMBINED_DATASET_DIR = DATASET_DIR / "artf" / "combined"
ARTF_CATEGORIES_DATASETS_DIR = DATASET_DIR / "artf" / "categories"


def download_artf_all_dataset(override_existing=False):
    # check if dataset already exists
    if os.path.exists(DATASET_DIR / "artf" / "combined") and not override_existing:
        print(
            f"Folder {DATASET_DIR / "artf" / "combined"} already exists, assuming dataset was already downloaded. If you want to redownload, set override_existing=True"
        )
        return

    # download dataset with all categories combined

    os.system(f"wget {ARTF_RESIZED_COMBINED_URL} -O {DATASET_DIR / 'artf_combined.zip'}")

    # extract dataset
    ARTF_COMBINED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    os.system(f"unzip {DATASET_DIR / 'artf_combined.zip'} -d {ARTF_COMBINED_DATASET_DIR}")


def download_artf_categories_dataset(override=False):
    # check if dataset already exists
    if os.path.exists(ARTF_CATEGORIES_DATASETS_DIR) and not override:
        print(
            f"Folder {ARTF_CATEGORIES_DATASETS_DIR} already exists, assuming dataset was already downloaded. If you want to redownload, set override_existing=True"
        )
        return

    if override and ARTF_CATEGORIES_DATASETS_DIR.exists():
        # remove the directory
        import shutil

        shutil.rmtree(str(ARTF_CATEGORIES_DATASETS_DIR))

    # this converts the dataset into tabular 'arrow' HF datasets and manually creates splits etc.
    # I just want to use HF as data storage for now so want the raw file layout from the dataset repo.

    # dataset = datasets.load_dataset(ARTF_RESIZED_CATEGORIES_HF_URL)
    # dataset.save_to_disk(ARTF_CATEGORIES_DATASETS_DIR)

    # so use this command instead:

    huggingface_hub.snapshot_download(
        ARTF_RESIZED_CATEGORIES_HF_URL, repo_type="dataset", local_dir=ARTF_CATEGORIES_DATASETS_DIR
    )


class ARTF_Tshirts_Dataset(DatasetContainer):
    json_train_path = ARTF_CATEGORIES_DATASETS_DIR / "tshirts-train_resized_512x256" / "tshirts-train.json"
    json_val_path = ARTF_CATEGORIES_DATASETS_DIR / "tshirts-val_resized_512x256" / "tshirts-val.json"
    json_test_path = ARTF_CATEGORIES_DATASETS_DIR / "tshirts-test_resized_512x256" / "tshirts-test.json"
    category_name = "tshirt"

    def download(override: bool = False):
        download_artf_categories_dataset(override=override)

class ARTF_Shorts_Dataset(DatasetContainer):
    json_train_path = ARTF_CATEGORIES_DATASETS_DIR / "shorts-train_resized_512x256" / "shorts-train.json"
    json_val_path = ARTF_CATEGORIES_DATASETS_DIR / "shorts-val_resized_512x256" / "shorts-val.json"
    json_test_path = ARTF_CATEGORIES_DATASETS_DIR / "shorts-test_resized_512x256" / "shorts-test.json"
    category_name = "shorts"

    def download(override: bool = False):
        download_artf_categories_dataset(override=override)

class ARTF_Towels_Dataset(DatasetContainer):
    json_train_path = ARTF_CATEGORIES_DATASETS_DIR / "towels-train_resized_512x256" / "towels-train.json"
    json_val_path = ARTF_CATEGORIES_DATASETS_DIR / "towels-val_resized_512x256" / "towels-val.json"
    json_test_path = ARTF_CATEGORIES_DATASETS_DIR / "towels-test_resized_512x256" / "towels-test.json"
    category_name = "towel"

    def download(override: bool = False):
        download_artf_categories_dataset(override=override)

if __name__ == "__main__":
    download_artf_categories_dataset(override=False)
