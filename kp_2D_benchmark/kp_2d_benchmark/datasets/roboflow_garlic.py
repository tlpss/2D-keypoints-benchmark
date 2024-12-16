"""https://universe.roboflow.com/gesture-recognition-dsn2n/garlic_keypoint"""


from kp_2d_benchmark import DATASET_DIR
from datasets import load_dataset

ROBOFLOW_GARLIC_DATASET_256_HF_REPO = "tlpss/roboflow-garlic-256x256"
ROBOFLOW_GARLIC_256_DIR = DATASET_DIR / "roboflow-garlic" /"256"

ROBOFLOW_GARLIC_DATASET_512_HF_REPO = "tlpss/roboflow-garlic-512x512"
ROBOFLOW_GARLIC_512_DIR = DATASET_DIR / "roboflow-garlic" /"512"


def download_roboflow_garlic_dataset_256_hf(override: bool = False):
    if not ROBOFLOW_GARLIC_256_DIR.exists() and not override:
        ROBOFLOW_GARLIC_256_DIR.mkdir(parents=True)
        dataset = load_dataset(ROBOFLOW_GARLIC_DATASET_256_HF_REPO)
        dataset.save_to_disk(ROBOFLOW_GARLIC_256_DIR)

def download_roboflow_garlic_dataset_512_hf(override: bool = False):
    if not ROBOFLOW_GARLIC_512_DIR.exists() and not override:
        ROBOFLOW_GARLIC_512_DIR.mkdir(parents=True)
        dataset = load_dataset(ROBOFLOW_GARLIC_DATASET_512_HF_REPO)
        dataset.save_to_disk(ROBOFLOW_GARLIC_512_DIR)


if __name__ == "__main__":
    load_roboflow_garlic_dataset_256_hf()