from kp_2d_benchmark.datasets import DATASETS

if __name__ == "__main__":
    for dataset in DATASETS:
        print(dataset)
        for split in ["train", "val", "test"]:
            print(f" dataset {dataset} - split {split} : {dataset.size(split)} images")
