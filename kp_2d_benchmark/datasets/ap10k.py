"""AP-10K: https://github.com/AlexTheBad/AP-10K (CC-BY-4.0)
Yu et al., "AP-10K: A Benchmark for Animal Pose Estimation in the Wild", NeurIPS 2021 Datasets & Benchmarks.

54 species from 23 families that all share the same 17-keypoint quadruped skeleton, which makes this the
only dataset here that varies appearance while holding the keypoint semantics fixed.

The upstream 54 species categories are collapsed into a single "animal" category and every instance is
cropped into its own image, see ap10k_to_coco.py. The upstream download only lives on google drive, so the
converted dataset is mirrored on huggingface.
"""

import huggingface_hub

from kp_2d_benchmark import DATASET_DIR
from kp_2d_benchmark.datasets.base import DatasetContainer

AP10K_DATASET_512_HF_REPO = "tlpss/ap10k-cropped-512x512"
AP10K_512_DIR = DATASET_DIR / "ap10k" / "512x512"


def download_ap10k_dataset_512_hf(override: bool = False):
    if not AP10K_512_DIR.exists() and not override:
        AP10K_512_DIR.mkdir(parents=True)
        huggingface_hub.snapshot_download(AP10K_DATASET_512_HF_REPO, repo_type="dataset", local_dir=AP10K_512_DIR)


class AP10K_512(DatasetContainer):
    json_train_path = AP10K_512_DIR / "train" / "annotations.json"
    json_val_path = AP10K_512_DIR / "val" / "annotations.json"
    json_test_path = AP10K_512_DIR / "test" / "annotations.json"
    category_name = "animal"

    def download(override: bool = False):
        download_ap10k_dataset_512_hf(override=override)


if __name__ == "__main__":
    download_ap10k_dataset_512_hf()
