from kp_2d_benchmark.datasets.roboflow_garlic import RoboflowGarlic256Dataset, RoboflowGarlic512Dataset

DATASETS = [RoboflowGarlic256Dataset, RoboflowGarlic512Dataset]

if __name__ == "__main__":
    for dataset in DATASETS:
        print(dataset)