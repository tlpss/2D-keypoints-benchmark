# 2D-keypoints-benchmark
A benchmark for 2D category-level keypoint detection with a strong focus on machine vision.


## Datasets sizes

| | image resolution | # train images | # val images | # test images |
|---|---|---|---|---|
RoboFlow Garlic | 256x256 | 697|104 |199 |
aRTF Tshirts |  512x256 |168 |42  | 400 |



## Performance numbers

**average keypoint distance**
| model      |   ARTF_Shorts_Dataset |   ARTF_Towels_Dataset |   ARTF_Tshirts_Dataset |   CUB200_2011_512 |   RoboflowGarlic256Dataset |
|:-----------|----------------------:|----------------------:|-----------------------:|------------------:|---------------------------:|
| pkd-maxvit |                  38.3 |                  17.3 |                   19   |              11.8 |                        6.9 |
| yolov8     |                  39   |                  24.5 |                   26.9 |              26   |                       24.2 |


## Local Development

### Local installation

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.