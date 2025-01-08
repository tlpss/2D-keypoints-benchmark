from kp_2d_benchmark.datasets.roboflow_garlic import RoboflowGarlic256Dataset, RoboflowGarlic512Dataset
from kp_2d_benchmark.datasets.artf import ARTF_Shorts_Dataset, ARTF_Tshirts_Dataset, ARTF_Towels_Dataset
DATASETS = [RoboflowGarlic256Dataset(), ARTF_Shorts_Dataset(), ARTF_Tshirts_Dataset(), ARTF_Towels_Dataset()]

if __name__ == "__main__":
    for dataset in DATASETS:
        print(dataset)