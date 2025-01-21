from kp_2d_benchmark import DATASET_DIR
from kp_2d_benchmark.datasets.base import DatasetContainer

CUB200_2011_512_DIR = DATASET_DIR / "CUB_200_2011" / "512x512"


class CUB200_2011_512(DatasetContainer):
    json_train_path = CUB200_2011_512_DIR / "train" / "train_annotations_train.json"
    json_val_path = CUB200_2011_512_DIR / "val" / "train_annotations_val.json"
    json_test_path = CUB200_2011_512_DIR / "test" / "test_annotations.json"
    category_name = "bird"

    def download(override: bool = False):
        raise NotImplementedError("Download not yet implemented for CUB200-2011-512")
