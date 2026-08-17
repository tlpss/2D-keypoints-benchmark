from kp_2d_benchmark.datasets.ap10k import AP10K_512
from kp_2d_benchmark.datasets.artf import ARTF_Shorts_Dataset, ARTF_Towels_Dataset, ARTF_Tshirts_Dataset
from kp_2d_benchmark.datasets.cub200 import CUB200_2011_512
from kp_2d_benchmark.datasets.gitw import GITW_256
from kp_2d_benchmark.datasets.roboflow_garlic import RoboflowGarlic256Dataset

DATASETS = [
    RoboflowGarlic256Dataset(),
    ARTF_Shorts_Dataset(),
    ARTF_Tshirts_Dataset(),
    ARTF_Towels_Dataset(),
    CUB200_2011_512(),
    AP10K_512(),
    GITW_256(),
]

if __name__ == "__main__":
    for dataset in DATASETS:
        print(dataset)
        print(dataset.num_keypoints)
