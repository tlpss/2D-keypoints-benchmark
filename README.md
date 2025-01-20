# 2D-keypoints-benchmark
A benchmark for 2D category-level keypoint detection with a strong focus on machine vision.


## Datasets sizes

| | image resolution | # train images | # val images | # test images |
|---|---|---|---|---|
RoboFlow Garlic | 256x256 | 697|104 |199 |
aRTF Tshirts |  512x256 |168 |42  | 400 |



## Performance numbers

**average keypoint distance**

| model      |   ARTF_Tshirts_Dataset |   RoboflowGarlic256Dataset |
|:-----------|-----------------------:|---------------------------:|
| pkd-maxvit |                   17.6 |                       10.8 |
| yolov8     |                   27   |                       24.2 |


## Local Development

### Local installation

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.