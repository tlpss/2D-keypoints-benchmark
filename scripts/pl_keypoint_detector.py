import json
from argparse import ArgumentParser
from pathlib import Path

import wandb
from keypoint_detection.tasks.train import train
from keypoint_detection.utils.path import get_wandb_log_dir_path
from tqdm import tqdm

from kp_2d_benchmark import DATA_DIR
from kp_2d_benchmark.datasets.base import DatasetContainer
from kp_2d_benchmark.eval.coco_results import COCOKeypointResult, COCOKeypointResults

COMMAND = "keypoint-detection train  --augment_train"


DEFAULT_DICT = {
    "keypoint_channel_configuration": None,
    "accelerator": "gpu",
    "ap_epoch_freq": 2,
    "check_val_every_n_epoch": 1,
    # "backbone_type": "DinoV2Linear",
    "backbone_type": "MaxVitUnet",
    "devices": 1,
    "early_stopping_relative_threshold": -1,
    "json_dataset_path": "",
    "json_test_dataset_path": "",
    "json_validation_dataset_path": "",
    "max_epochs": 150,  # TOOD: max steps instead of max keypoints?
    "maximal_gt_keypoint_pixel_distances": "2 4 8",
    "minimal_keypoint_extraction_pixel_distance": 4,
    "precision": 16,
    "seed": 2024,
    "heatmap_sigma": 2,
    "learning_rate": 0.0002,
    "batch_size": 8,
    ###
    "wandb_entity": None,
    "wandb_project": "kp-benchmark",
    "wandb_name": None,
    "augment_train": True,
}


def train_dector_from_dict(arg_dict):
    def get_argparse_defaults(parser):
        defaults = {}
        for action in parser._actions:
            if not action.required and action.dest != "help":
                defaults[action.dest] = action.default
        return defaults

    from keypoint_detection.tasks.train import (
        BackboneFactory,
        KeypointDetector,
        KeypointsDataModule,
        Trainer,
        add_system_args,
    )

    parser = ArgumentParser()
    parser = add_system_args(parser)
    parser = KeypointDetector.add_model_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = KeypointsDataModule.add_argparse_args(parser)
    parser = BackboneFactory.add_to_argparse(parser)

    # get parser arguments and filter the specified arguments
    defaults = get_argparse_defaults(parser)
    hparams = defaults

    hparams.update(arg_dict)
    model, trainer = train(hparams)
    return model, trainer


def create_coco_results_file(dataset: DatasetContainer, model, results_path):
    import datetime

    import cv2
    import torch

    # load the dataset
    with open(dataset.json_test_path) as f:
        import json

        data = json.load(f)

    results = []

    # load_from_checkpoint returns a cpu model, and the input tensors follow model.device, so without this
    # the whole test set is processed on cpu (~0.3s per image instead of ~0.006s).
    if torch.cuda.is_available():
        model.to("cuda")

    # model to eval mode
    model.eval()

    # start timer
    start = datetime.datetime.now()

    for image in tqdm(data["images"]):
        # create absolute path
        image_path = Path(dataset.json_test_path).parent / image["file_name"]
        # load the image. cv2 reads BGR while training loads RGB (keypoint_detection uses skimage.io.imread),
        # so convert here to match what the model was trained on.
        img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        # convert to torch tensor
        img = torch.from_numpy(img).to(model.device).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            # get the predictions
            pred = model(img)
            from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool

            keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(pred, max_keypoints=1, return_scores=True)
        keypoints = keypoints[0]
        scores = scores[0]
        # if not a single keypoint was detected -> skip
        if len(scores) == 0:
            continue
        # get best keypoint per channel
        # if none was detected, set keypoint to center of the image.
        final_keypoints = []
        final_confidences = []
        for i in range(len(keypoints)):
            channel_keypoints = keypoints[i]
            channel_scores = scores[i]
            if len(channel_keypoints) == 0:
                final_keypoints.extend([img.shape[3] // 2, img.shape[2] // 2, 0])
                final_confidences.append(0)
            else:
                best_keypoint = channel_keypoints[channel_scores.index(max(channel_scores))]
                final_keypoints.extend([best_keypoint[0], best_keypoint[1], 2])
                final_confidences.append(max(channel_scores))

        keypoints = final_keypoints
        confidences = final_confidences
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


def train_and_test(train_name, dataset: DatasetContainer):
    from keypoint_detection.utils.load_checkpoints import load_from_checkpoint

    arg_dict = DEFAULT_DICT.copy()
    arg_dict["json_dataset_path"] = dataset.json_train_path
    arg_dict["json_validation_dataset_path"] = dataset.json_val_path
    # arg_dict["json_test_dataset_path"] = dataset.json_test_path
    arg_dict["wandb_name"] = train_name

    categories = json.load(open(dataset.json_train_path))["categories"]
    keypoints = None
    for category in categories:
        if category["name"] == dataset.category_name:
            keypoints = category["keypoints"]
    assert keypoints is not None, f"Category {dataset.category_name} not found in dataset"
    channel_config = keypoints
    arg_dict["keypoint_channel_configuration"] = channel_config

    if "cub" in (dataset.__repr__().lower()):
        arg_dict["max_epochs"] = 40
        # TODO: specify # steps instead of # epochs
    if "ap10k" in (dataset.__repr__().lower()):
        # ~9.1k train samples vs ~5.4k for cub, scaled down to keep the number of steps comparable.
        arg_dict["max_epochs"] = 24

    wandb.init(
        name=arg_dict["wandb_name"],
        project=arg_dict["wandb_project"],
        entity=arg_dict["wandb_entity"],
        config=arg_dict,
        dir=get_wandb_log_dir_path(),  # dir should already exist! will fallback to /tmp and not log images otherwise..
    )

    model, trainer = train_dector_from_dict(arg_dict)
    # get the best checkpoint from the trainer
    ckpt_path = trainer.checkpoint_callback.best_model_path
    # load that checkpoint into the model
    model = load_from_checkpoint(ckpt_path)
    model.eval()

    results_path = DATA_DIR / "results" / f"model=pkd-{arg_dict['backbone_type']},dataset={dataset.__repr__()}.json"
    create_coco_results_file(dataset, model, results_path)

    wandb.finish()


if __name__ == "__main__":
    from kp_2d_benchmark.datasets import DATASETS

    # dataset = RoboflowGarlic256Dataset()
    # train_name = "pkd-maxvit-roboflow_garlic256"
    # dataset = CUB200_2011_512()
    # train_name = "pkd-dinov2-cub200"
    # train_and_test(train_name,dataset)
    # dataset.download()

    for dataset in DATASETS:
        train_name = f"pkd-dinov2-{dataset.__repr__()}"
        train_and_test(train_name, dataset)
