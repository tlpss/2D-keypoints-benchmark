from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess

from kp_2d_benchmark import DATA_DIR
from kp_2d_benchmark.datasets.base import DatasetContainer
from kp_2d_benchmark.eval.coco_results import COCOKeypointResult, COCOKeypointResults, CocoKeypointsDataset

from keypoint_detection.tasks.train import train
COMMAND = "keypoint-detection train  --augment_train"


DEFAULT_DICT = {
    "keypoint_channel_configuration": None,
    "accelerator": "gpu",
    "ap_epoch_freq": 1,
    "check_val_every_n_epoch": 1,
    "backbone_type": "MaxVitUnet",
    "devices": 1,
    "early_stopping_relative_threshold": -1,
    "json_dataset_path": "",
    "json_test_dataset_path": "",
    "json_validation_dataset_path": "",
    "max_epochs": 50,
    "maximal_gt_keypoint_pixel_distances": "4 8 16",  
    "minimal_keypoint_extraction_pixel_distance": 4,
    "precision": 16,
    "seed": 2024,
    "heatmap_sigma": 4,
    "learning_rate": 0.0003,
    "batch_size": 8,
    ###
    # "wandb_entity": "tlips",
    "wandb_project": "kp-benchmark",
    "wandb_name": None,
}


def train_dector_from_dict(arg_dict):
    def get_argparse_defaults(parser):
        defaults = {}
        for action in parser._actions:
            if not action.required and action.dest != "help":
                defaults[action.dest] = action.default
        return defaults
    
    from keypoint_detection.tasks.train import add_system_args, KeypointDetector, Trainer, KeypointsDataModule, BackboneFactory
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



        with torch.no_grad():
            # get the predictions
            pred = model(img)
            from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
            keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(pred,max_keypoints=1,return_scores=True)
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
                final_keypoints.extend([img.shape[3]//2, img.shape[2]//2, 0])
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



def train_and_test(train_name, dataset: DatasetContainer):
    from keypoint_detection.utils.load_checkpoints import load_from_checkpoint

    arg_dict = DEFAULT_DICT.copy()
    arg_dict["json_dataset_path"] = dataset.json_train_path
    arg_dict["json_validation_dataset_path"] = dataset.json_val_path
    #arg_dict["json_test_dataset_path"] = dataset.json_test_path
    arg_dict["wandb_name"] = train_name

    categories = json.load(open(dataset.json_train_path))["categories"]
    keypoints = None
    for category in categories:
        if category["name"] == dataset.category_name:
            keypoints = category["keypoints"]
    assert keypoints is not None, f"Category {dataset.category_name} not found in dataset"
    channel_config = keypoints
    arg_dict["keypoint_channel_configuration"] = channel_config
    
    model, trainer = train_dector_from_dict(arg_dict)
    # get the best checkpoint from the trainer
    ckpt_path = trainer.checkpoint_callback.best_model_path
    # load that checkpoint into the model
    model  = load_from_checkpoint(ckpt_path)
    model.eval()

    results_path = DATA_DIR / "results" / f"{train_name}_results.json"
    create_coco_results_file(dataset, model, results_path)

if __name__ == "__main__":
    from kp_2d_benchmark.datasets.roboflow_garlic import RoboflowGarlic256Dataset
    from kp_2d_benchmark.datasets.artf import ARTF_Shorts_Dataset, ARTF_Tshirts_Dataset, ARTF_Towels_Dataset
    
    # dataset = RoboflowGarlic256Dataset()
    # train_name = "pkd-maxvit-roboflow_garlic256"

    dataset = ARTF_Tshirts_Dataset()
    train_name = "pkd-maxvit-artf_tshirts"

    # dataset.download()

    train_and_test(train_name, dataset)

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

    
    
    