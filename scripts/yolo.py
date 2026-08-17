# for each dataset

# train model

# create coco results file

from datetime import datetime
from pathlib import Path

import wandb
from airo_dataset_tools.coco_tools.coco_instances_to_yolo import create_yolo_dataset_from_coco_instances_dataset
from ultralytics import YOLO, settings

from kp_2d_benchmark import DATA_DIR

# store results file
from kp_2d_benchmark.datasets.base import DatasetContainer
from kp_2d_benchmark.eval.coco_results import COCOKeypointResult, COCOKeypointResults

YOLO_LOG_DIR = DATA_DIR / "runs" / "yolo"
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
    import datetime

    import cv2
    import torch

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
        result = COCOKeypointResult(
            image_id=image["id"],
            category_id=category_id,
            keypoints=keypoints,
            score=sum(confidences) / len(confidences),
            per_keypoint_scores=confidences,
        )
        results.append(result)

    end = datetime.datetime.now()
    print(f"Time taken: {end-start}")
    # print time per image
    print(f"Time per image: {(end-start)/len(data['images'])}")

    # save the results
    results = COCOKeypointResults(root=results)
    with open(results_path, "w") as f:
        f.write(results.model_dump_json())


def train_and_test_yolo_keypoints(
    train_name, dataset: DatasetContainer, model_name: str = "yolov8s-pose", model_label: str = None
):
    """Train an ultralytics pose model on a benchmark dataset and write its coco results file.

    model_name: any ultralytics pose checkpoint, e.g. "yolov8s-pose" or "yolo26s-pose".
    model_label: name used in the results file and hence in metrics.csv. Defaults to the checkpoint name
        without its "-pose" suffix. Passed explicitly for yolov8, whose column predates this argument.
    """
    model_label = model_label or model_name.removesuffix("-pose")

    wandb.init(project="kp-benchmark", name=train_name)

    # disable wandb finish to keep ultlralytics from finishing the run
    WANDB_FINISH = wandb.run.finish
    wandb.run.finish = lambda: None

    # append wandb run id to train_name to make it unique and avoid suffix by ultralytics
    yolo_train_name = f"{train_name}_{wandb.run.id}"

    # create a model pretrained on COCO
    model = YOLO(model_name)

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

    create_yolo_kp_data_yaml(
        train_yolo_dataset_path, val_yolo_dataset_path, dataset.category_name, dataset.num_keypoints, FILENAME
    )

    # get the image size from the dataset
    with open(dataset.json_train_path) as f:
        import json

        data = json.load(f)
        img_size = data["images"][0]["width"]
    print("training")
    model.train(data=FILENAME, epochs=100, imgsz=img_size, name=yolo_train_name)

    # evaluate the model
    # load best checkpoint. ask the trainer where it saved instead of assuming YOLO_LOG_DIR:
    # ultralytics does not always honour the runs_dir setting and then writes to ./runs.
    best_checkpoint_path = model.trainer.best
    print("Evaluating model")
    model = YOLO(best_checkpoint_path)

    # set test dataset as val to evaluate
    create_yolo_kp_data_yaml(
        train_yolo_dataset_path, test_yolo_dataset_path, dataset.category_name, dataset.num_keypoints, FILENAME
    )
    test_results = model.val(data=FILENAME)
    all_aps = test_results.pose.all_ap
    m_ap = test_results.pose.map

    if wandb.run:
        wandb.log({"test/pose_mAP": m_ap})
        wandb.log({"test/bbox_mAP": test_results.box.map})

    print(f"mAP: {m_ap}")
    print("all APs")
    print(all_aps)

    # create coco results file
    results_path = DATA_DIR / "results" / f"model={model_label},dataset={dataset.__repr__()}.json"
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
    from argparse import ArgumentParser

    import kp_2d_benchmark.datasets as datasets

    parser = ArgumentParser(description="train an ultralytics pose model on one benchmark dataset")
    parser.add_argument("dataset", help="name of a DatasetContainer in kp_2d_benchmark.datasets, e.g. AP10K_512")
    parser.add_argument("--model", default="yolov8s-pose", help="ultralytics pose checkpoint to finetune")
    parser.add_argument(
        "--label", default=None, help="name to use in metrics.csv, defaults to the checkpoint without '-pose'"
    )
    args = parser.parse_args()

    label = args.label or args.model.removesuffix("-pose")
    train_and_test_yolo_keypoints(
        f"{label}-{args.dataset}", getattr(datasets, args.dataset)(), model_name=args.model, model_label=args.label
    )
