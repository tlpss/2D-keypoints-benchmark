# for each dataset 

# train model 

# create coco results file

# store results file 
from kp_2d_benchmark.datasets import DATASETS
from kp_2d_benchmark.datasets.base import DatasetContainer

from kp_2d_benchmark import DATA_DIR
    
from datetime import datetime
from pathlib import Path

import wandb
from ultralytics import YOLO, settings

from airo_dataset_tools.coco_tools.coco_instances_to_yolo import create_yolo_dataset_from_coco_instances_dataset
from kp_2d_benchmark.eval.coco_results import COCOKeypointResults, COCOKeypointResult


YOLO_LOG_DIR = DATA_DIR  / "runs" / "yolo"
YOLO_DATASET_DIR = DATA_DIR / "yolo"
settings.update({"datasets_dir": str(YOLO_DATASET_DIR)})
# rundir
settings.update({"runs_dir": str(YOLO_LOG_DIR)})
settings.update({"weights_dir": str(YOLO_LOG_DIR)})



# create temp yolo data.yaml file
def create_yolo_kp_data_yaml(train_dataset_path, val_dataset_path, class_name, num_keypoints, filename):

    train_dataset_path = Path(train_dataset_path)
    val_dataset_path = Path(val_dataset_path)
    for path in [train_dataset_path, val_dataset_path]:
        if path is not None:
            # if absolute path, convert to relative path
            if path.is_absolute():
                path = path.relative_to(DATA_DIR)

    data = f"""
    path: .
    train: {str(train_dataset_path)}
    val: {str(val_dataset_path)}

    kpt_shape: [{num_keypoints},3]
    names:
        0: {class_name}
    """
    # ^ allows to select single category out of multiple category datasets.

    with open(filename, "w") as f:
        f.write(data)
    print()


def create_coco_results_file(dataset: DatasetContainer, model, results_path):
    import cv2
    import torch
    import datetime

    # load the dataset
    with open(dataset.json_test_path) as f:
        import json
        data = json.load(f)
    
    results = []

    # model to eval mode
    model.eval()

    # start timer
    start = datetime.datetime.now()

    for image in data["images"]:
        # create absolute path
        image_path = Path(dataset.json_test_path).parent / image["file_name"]
        # load the image
        img = cv2.imread(str(image_path))
        # convert to torch tensor
        img = torch.from_numpy(img).to(model.device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        # get the predictions
        pred = model(img)

        # for each channel (first dimension),
        # get the keypoint with max confidence
        boxes = pred[0].boxes
        if len(boxes) == 0:
            continue
        highest_confidence_box_index = torch.argmax(boxes.conf)

        keypoint_object = pred[0].keypoints
        keypoints = keypoint_object.xy[highest_confidence_box_index]
        keypoints = keypoints.cpu().tolist()

        for kp in keypoints:
            kp.append(2)
        
        # flatten the nested list
        keypoints = [item for sublist in keypoints for item in sublist]

        confidences = keypoint_object.conf[highest_confidence_box_index]
        confidences = confidences.cpu().tolist()


        # find the id of the category
        category_id = None
        for category in data["categories"]:
            if category["name"] == dataset.category_name:
                category_id = category["id"]
        if category_id is None:
            raise ValueError("Category not found in the dataset")
        result = COCOKeypointResult(image_id=image["id"], category_id=category_id, keypoints=keypoints, score=sum(confidences)/len(confidences), per_keypoint_scores=confidences)
        results.append(result)
            
    end = datetime.datetime.now()
    print(f"Time taken: {end-start}")
    # print time per image
    print(f"Time per image: {(end-start)/len(data['images'])}")

    # save the results
    results = COCOKeypointResults(root=results)
    with open(results_path, "w") as f:
        f.write(results.model_dump_json())


def train_and_test_yolo_keypoints(train_name, dataset: DatasetContainer):

    wandb.init(project="kp-benchmark", name=train_name)

    # disable wandb finish to keep ultlralytics from finishing the run
    WANDB_FINISH = wandb.run.finish
    wandb.run.finish = lambda: None


    # append wandb run id to train_name to make it unique and avoid suffix by ultralytics
    yolo_train_name = f"{train_name}_{wandb.run.id}"

    # create a model pretrained on COCO
    model = YOLO("yolov8s-pose")

    # create the temp yolo dataset

    DATASET_PATH = YOLO_DATASET_DIR / train_name

    train_yolo_dataset_path = DATASET_PATH / "train"
    val_yolo_dataset_path = DATASET_PATH / "val"
    test_yolo_dataset_path = DATASET_PATH / "test"

    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    train_yolo_dataset_path.mkdir(parents=True, exist_ok=True)
    val_yolo_dataset_path.mkdir(parents=True, exist_ok=True)
    test_yolo_dataset_path.mkdir(parents=True, exist_ok=True)


    create_yolo_dataset_from_coco_instances_dataset(dataset.json_train_path, str(DATASET_PATH / "train"))
    create_yolo_dataset_from_coco_instances_dataset(dataset.json_val_path, str(DATASET_PATH / "val"))
    create_yolo_dataset_from_coco_instances_dataset(dataset.json_test_path, str(DATASET_PATH / "test"))

    

    # create temp yolo data.yaml file
    FILENAME = f"{datetime.now()}_data.yaml"
    create_yolo_kp_data_yaml(train_yolo_dataset_path, val_yolo_dataset_path, dataset.category_name,dataset.num_keypoints, FILENAME)

    # get the image size from the dataset
    with open(dataset.json_train_path) as f:
        import json 
        data = json.load(f)
        img_size = data["images"][0]["width"]
    model.train(data=FILENAME, epochs=100, imgsz=img_size, name=yolo_train_name)

    # evaluate the model
    # load best checkpoint
    model = YOLO(f"{YOLO_LOG_DIR}/pose/{yolo_train_name}/weights/best.pt")

    # set test dataset as val to evaluate
    create_yolo_kp_data_yaml(train_yolo_dataset_path, test_yolo_dataset_path, dataset.category_name, dataset.num_keypoints, FILENAME)
    test_results = model.val(data=FILENAME)
    all_aps = test_results.pose.all_ap
    m_ap = test_results.pose.map

    if wandb.run:
        pass
        wandb.log({"test/pose_mAP": m_ap})
        wandb.log({"test/bbox_mAP": test_results.box.map})

    print(f"mAP: {m_ap}")
    print("all APs")
    print(all_aps)


    # create coco results file
    results_path = DATA_DIR / "results" / f"{train_name}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    create_coco_results_file(dataset, model, results_path)

    # remove the temp yolo data.yaml file
    import os
    os.remove(FILENAME)

    # remove the temp yolo datasets
    import shutil
    shutil.rmtree(DATASET_PATH)


    WANDB_FINISH()

if __name__ == "__main__":
    from kp_2d_benchmark.datasets import RoboflowGarlic256Dataset, ARTF_Tshirts_Dataset
    import json 
    from kp_2d_benchmark.eval.coco_results import CocoKeypointsDataset
    
    # dataset = RoboflowGarlic256Dataset()
    # train_name = "yolov8-roboflow_garlic256"

    dataset = ARTF_Tshirts_Dataset()
    train_name = "yolov8-artf_tshirts"
    train_and_test_yolo_keypoints(train_name, dataset)

    from kp_2d_benchmark.eval.calculate_keypoint_distance_metrics import calculate_keypoint_distances,calculate_average_distances

    results_path = DATA_DIR / "results" / f"{train_name}_results.json"
    results = COCOKeypointResults(json.load(open(results_path)))

    coco_dataset = CocoKeypointsDataset(**json.load(open(dataset.json_test_path)))
    distance_dict = calculate_keypoint_distances(coco_dataset, results)
    average_distance_dict = calculate_average_distances(distance_dict)
    
    print(train_name)
    print(average_distance_dict)
    
    key = list(average_distance_dict.keys())[0]
    avg_distances = list(average_distance_dict[key].values())
    print(avg_distances)
    print(f"Average distance: {sum(avg_distances)/len(avg_distances)}")

    

    

