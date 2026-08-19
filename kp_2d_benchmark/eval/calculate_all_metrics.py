"""Recompute every metric of the benchmark from the result files and write metrics.csv.

Result files are named `data/results/model=<label>,dataset=<DatasetRepr>.json`; that filename is the
schema, the model and dataset are parsed out of it. Subdirectories are skipped, which is how superseded
results are archived.

The metrics themselves are defined in the "Metrics" section of the README.
"""

import csv
import json
import os
from typing import Callable, Dict, List

from kp_2d_benchmark.datasets import DATASETS
from kp_2d_benchmark.eval.calculate_keypoint_ap import AP_ALPHA, calculate_keypoint_ap
from kp_2d_benchmark.eval.calculate_keypoint_distance_metrics import (
    calculate_average_distances,
    calculate_detection_rate,
    calculate_keypoint_distances,
    calculate_median_distances,
    calculate_median_normalized_distance,
    calculate_pck_at_alpha,
    calculate_strict_success_rate,
)
from kp_2d_benchmark.eval.coco_results import COCOKeypointResults

PCK_ALPHA = 0.05
STRICT_SUCCESS_ALPHA = 0.05


def _percentage(value: float) -> str:
    return f"{100 * value:.1f}"


def _three_decimals(value: float) -> str:
    return f"{value:.3f}"


def _one_decimal(value: float) -> str:
    return f"{value:.1f}"


# the five metrics the README reports, in the order they appear there, and how to format each of them.
# the raw pixel distances are kept in the csv for continuity but are no longer headline numbers.
HEADLINE_METRICS: Dict[str, Callable[[float], str]] = {
    "detection_rate": _percentage,
    "median_nme": _three_decimals,
    f"pck@{PCK_ALPHA}": _percentage,
    f"strict_success@{STRICT_SUCCESS_ALPHA}": _percentage,
    f"mAP@{AP_ALPHA}": _three_decimals,
}

LEGACY_METRICS: Dict[str, Callable[[float], str]] = {
    "average_keypoint_distance": _one_decimal,
    "median_keypoint_distance": _one_decimal,
}

METRIC_FORMATTERS = {**HEADLINE_METRICS, **LEGACY_METRICS}

CSV_FIELDS = ["model", "dataset", *METRIC_FORMATTERS]


def parse_result_file_name(file_path: str):
    """Parse the model label and dataset repr out of a `model=<>,dataset=<>.json` file name."""
    file_name = os.path.basename(file_path)
    model, dataset_name = (
        file_name.split(".json")[0].split(",")[0].split("=")[1],
        file_name.split(".json")[0].split(",")[1].split("=")[1],
    )
    return model, dataset_name


def get_metrics(file_path: str) -> Dict[str, object]:
    """All metrics for a single result file, as a row for metrics.csv."""
    model, dataset_name = parse_result_file_name(file_path)

    dataset = None
    for x in DATASETS:
        if x.__repr__() == dataset_name:
            dataset = x
    if not dataset:
        raise ValueError(f"Dataset {dataset_name} not found in DATASETS")

    with open(file_path, "r") as f:
        results = COCOKeypointResults(json.load(f))

    test_dataset = dataset.get_split("test")
    distance_dict = calculate_keypoint_distances(test_dataset, results)

    average_distance_dict = calculate_average_distances(distance_dict)
    median_distance_dict = calculate_median_distances(distance_dict)
    avg_distance = sum([sum(x.values()) for x in average_distance_dict.values()]) / sum(
        [len(x) for x in average_distance_dict.values()]
    )
    median_distance = sum([sum(x.values()) for x in median_distance_dict.values()]) / sum(
        [len(x) for x in median_distance_dict.values()]
    )

    ap = calculate_keypoint_ap(test_dataset, results, alpha=AP_ALPHA)

    return {
        "model": model,
        "dataset": dataset_name,
        "detection_rate": calculate_detection_rate(distance_dict),
        "median_nme": calculate_median_normalized_distance(distance_dict),
        f"pck@{PCK_ALPHA}": calculate_pck_at_alpha(distance_dict, alpha=PCK_ALPHA),
        f"strict_success@{STRICT_SUCCESS_ALPHA}": calculate_strict_success_rate(
            distance_dict, alpha=STRICT_SUCCESS_ALPHA
        ),
        f"mAP@{AP_ALPHA}": ap["mAP"],
        "average_keypoint_distance": avg_distance,
        "median_keypoint_distance": median_distance,
    }


def write_metrics_csv(rows: List[Dict[str, object]], metric_csv_path: str) -> None:
    with open(metric_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_results_csv_as_markdown_table(metric_csv_path: str, metrics: Dict[str, Callable] = None) -> str:
    """Render one markdown table per metric, models in rows and datasets in columns."""
    import pandas as pd

    metrics = metrics if metrics is not None else HEADLINE_METRICS

    df = pd.read_csv(metric_csv_path)
    tables = []
    for metric, formatter in metrics.items():
        table = df.pivot(index="model", columns="dataset", values=metric)
        table = table.map(formatter)
        # disable_numparse keeps tabulate from re-parsing the formatted strings back into numbers, which
        # would drop trailing zeroes and undo the per-metric formatting. that also loses the numeric
        # alignment, so the columns are aligned explicitly.
        alignment = ("left",) + ("right",) * len(table.columns)
        tables.append(f"**{metric}**\n\n{table.to_markdown(disable_numparse=True, colalign=alignment)}")
    return "\n\n".join(tables)


if __name__ == "__main__":
    from kp_2d_benchmark import DATA_DIR

    results_dir = DATA_DIR / "results"
    metric_csv_path = "metrics.csv"

    rows = []
    for file in sorted(os.listdir(results_dir)):
        if file.endswith(".json"):
            file = os.path.join(results_dir, file)
            rows.append(get_metrics(file))
            print(f"metrics for {os.path.basename(file)} done")

    write_metrics_csv(rows, metric_csv_path)
    print(f"Metrics for {len(rows)} result files saved to {metric_csv_path}\n")
    print(format_results_csv_as_markdown_table(metric_csv_path))
