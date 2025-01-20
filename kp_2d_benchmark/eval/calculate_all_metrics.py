import json

from kp_2d_benchmark.datasets import DATASETS
from kp_2d_benchmark.eval.calculate_keypoint_distance_metrics import (
    calculate_average_distances,
    calculate_keypoint_distances,
)
from kp_2d_benchmark.eval.coco_results import COCOKeypointResults


def get_metrics(file_path, metric_csv_path):
    # parse file_path: model=<>,dataset=<>.json
    model, dataset_name = (
        file_path.split(".json")[0].split(",")[0].split("=")[1],
        file_path.split(".json")[0].split(",")[1].split("=")[1],
    )

    # get model & dataset name
    dataset = None
    for x in DATASETS:
        if x.__repr__() == dataset_name:
            dataset = x
    if not dataset:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS")

    results = COCOKeypointResults(json.load(open(file_path)))

    # calculate all metrics.
    test_dataset = dataset.get_split("test")
    distance_dict = calculate_keypoint_distances(test_dataset, results)
    average_distance_dict = calculate_average_distances(distance_dict)
    print(average_distance_dict)
    avg_distance = sum([sum(x.values()) for x in average_distance_dict.values()]) / sum(
        [len(x) for x in average_distance_dict.values()]
    )

    # load the csv file and append the new metrics
    with open(metric_csv_path, "r") as f:
        data = f.readlines()
    data.append(f"{model},{dataset_name},{avg_distance}\n")
    with open(metric_csv_path, "w") as f:
        f.writelines(data)
    print(f"Metrics for model {model} and dataset {dataset_name} saved to {metric_csv_path}")


def format_results_cvs_as_markdown_table(metric_csv_path):
    # create pandas table for csv
    import pandas as pd

    df = pd.read_csv(metric_csv_path)
    # format floats to two decimal places
    for metric in df.columns[2:]:
        df[metric] = df[metric].apply(lambda x: f"{x:.1f}")
    # create a table for each metric with models in rows and values in columns
    for metric in df.columns[2:]:
        print(metric)
        table = df.pivot(index="model", columns="dataset", values=metric)
        print(table.to_markdown())


if __name__ == "__main__":
    import os

    from kp_2d_benchmark import DATA_DIR

    results_dir = DATA_DIR / "results"
    metric_csv_path = "metrics.csv"

    with open(metric_csv_path, "w") as f:
        f.write("model,dataset,average_keypoint_distance\n")
    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            # get full path
            file = os.path.join(results_dir, file)
            get_metrics(file, metric_csv_path)

    print(format_results_cvs_as_markdown_table(metric_csv_path))
