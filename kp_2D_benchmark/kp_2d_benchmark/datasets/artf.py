""" 
artf dataset 

"""
from kp_2d_benchmark import DATASET_DIR
import os

ARTF_RESIZED_SPLITS_ZENODO_URL = "https://zenodo.org/records/10875542/files/aRTFClothes-resized-paper-splits.zip?download=1"
ARTF_RESIZED_COMBINED_URL = "https://zenodo.org/records/14501499/files/artf-all-categories-resized_512x256.zip?download=1"

def download_dataset(override_existing=False):
    # check if dataset already exists
    if os.path.exists(DATASET_DIR / "artf") and not override_existing:
        print(f"Folder {DATASET_DIR / "artf"} already exists, assuming dataset was already downloaded. If you want to redownload, set override_existing=True")
        return

    # download dataset with all categories combined

    os.system(f"wget {ARTF_RESIZED_COMBINED_URL} -O {DATASET_DIR / 'artf.zip'}")

    # extract dataset

    os.system(f"unzip {DATASET_DIR / 'artf.zip'} -d {DATASET_DIR / 'artf'}")
    
if __name__ =="__main__":
    download_dataset()