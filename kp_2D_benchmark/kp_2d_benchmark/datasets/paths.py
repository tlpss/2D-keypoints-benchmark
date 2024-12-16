from kp_2d_benchmark import DATASET_DIR

ARFT_DATASET_TRAIN_JSON_PATH = DATASET_DIR / "artf" / "artf-train_resized_512"/ "annotations.json"
ARFT_DATASET_VAL_JSON_PATH = DATASET_DIR / "artf" / "artf-val_resized_512" / "annotations.json"
ARFT_DATASET_TEST_JSON_PATH = DATASET_DIR / "artf" / "artf-test_resized_512" / "annotations.json"


if __name__ == "__main__":
    # assert all paths exist
    
    # gather all variables in the file that have json extension
    json_paths = [v for k, v in locals().items() if isinstance(v, str) and v.endswith(".json")]
    for json_path in json_paths:
        assert os.path.exists(json_path), f"Path {json_path} does not exist"