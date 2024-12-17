from kp_2d_benchmark import DATASET_DIR
from kp_2d_benchmark.datasets.artf import ARTF_CATEGORIES_DATASETS_DIR

ARTF_DATASET_TRAIN_JSON_PATH = DATASET_DIR / "artf" / "artf-train_resized_512" / "annotations.json"
ARTF_DATASET_VAL_JSON_PATH = DATASET_DIR / "artf" / "artf-val_resized_512" / "annotations.json"
ARTF_DATASET_TEST_JSON_PATH = DATASET_DIR / "artf" / "artf-test_resized_512" / "annotations.json"

ARTF_TSHIRTS_DATASET_TRAIN_JSON_PATH = (
    ARTF_CATEGORIES_DATASETS_DIR / "tshirts-train_resized_512x256" / "tshirts-train.json"
)
ARTF_TSHIRTS_DATASET_VAL_JSON_PATH = ARTF_CATEGORIES_DATASETS_DIR / "tshirts-val_resized_512x256" / "tshirts-val.json"
ARTF_TSHIRTS_DATASET_TEST_JSON_PATH = ...


ROBOFLOW_GARLIC_256_DATASET_TRAIN_JSON_PATH = DATASET_DIR / "roboflow-garlic" / "256" / "train" / "annotations.json"
ROBOFLOW_GARLIC_256_DATASET_VAL_JSON_PATH = DATASET_DIR / "roboflow-garlic" / "256" / "val" / "annotations.json"
ROBOFLOW_GARLIC_256_DATASET_TEST_JSON_PATH = DATASET_DIR / "roboflow-garlic" / "256" / "test" / "annotations.json"

if __name__ == "__main__":
    import json
    import pathlib

    # gather all variables in the file that have json extension
    print("checking all paths")
    json_paths = [v for k, v in locals().items() if isinstance(v, pathlib.Path) and v.name.endswith("json")]

    for json_path in json_paths:
        if not json_path.exists():
            print(f"path does not exist: {json_path.relative_to(DATASET_DIR)}")
            continue
        json_dict = json.load(open(json_path, "r"))
        print(f"{json_path.relative_to(DATASET_DIR)} contains {len(json_dict['annotations'])} images")
