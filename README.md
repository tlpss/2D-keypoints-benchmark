# 2D-keypoints-benchmark
A benchmark for 2D category-level keypoint detection with a strong focus on machine vision.



## Performance numbers
|  | RoboFlow Garlic 256x256 | aRTF Tshirts | aRTF Shorts 512 | aRTF towels |
| --- | --- | --- | ---|  --- |
| YOLOv8 |24.2 |27.0 | x| x |
| PKD - MaxViT | 10.8 | 28.2 | x | x|


## Local Development

### Local installation

- clone this repo
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.