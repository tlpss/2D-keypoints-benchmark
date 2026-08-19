"""Recompute every metric of the benchmark from the result files and write metrics.csv.

Result files are named `data/results/model=<label>,dataset=<DatasetRepr>.json`; that filename is the
schema, the model and dataset are parsed out of it. Subdirectories are skipped, which is how superseded
results are archived.

The metrics themselves are defined in the "Metrics" section of the README.
"""

import csv
import json
import os
from typing import Callable, Dict, List, NamedTuple

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


class MetricSpec(NamedTuple):
    """How to render a metric, and which end of its range is good."""

    format: Callable[[float], str]
    higher_is_better: bool


# the five metrics the README reports, in the order they appear there.
# the raw pixel distances are kept in the csv for continuity but are no longer headline numbers.
HEADLINE_METRICS: Dict[str, MetricSpec] = {
    "detection_rate": MetricSpec(_percentage, higher_is_better=True),
    "median_nme": MetricSpec(_three_decimals, higher_is_better=False),
    f"pck@{PCK_ALPHA}": MetricSpec(_percentage, higher_is_better=True),
    f"strict_success@{STRICT_SUCCESS_ALPHA}": MetricSpec(_percentage, higher_is_better=True),
    f"mAP@{AP_ALPHA}": MetricSpec(_three_decimals, higher_is_better=True),
}

LEGACY_METRICS: Dict[str, MetricSpec] = {
    "average_keypoint_distance": MetricSpec(_one_decimal, higher_is_better=False),
    "median_keypoint_distance": MetricSpec(_one_decimal, higher_is_better=False),
}

METRIC_FORMATTERS = {**HEADLINE_METRICS, **LEGACY_METRICS}

# short column labels, so that a table with every dataset in it still fits on a page. datasets that are
# not listed keep their repr, so adding one degrades the layout rather than breaking the table.
DATASET_LABELS = {
    "RoboflowGarlic256Dataset": "Garlic",
    "ARTF_Towels_Dataset": "Towels",
    "ARTF_Shorts_Dataset": "Shorts",
    "ARTF_Tshirts_Dataset": "Tshirts",
    "CUB200_2011_512": "CUB-200",
    "AP10K_512": "AP-10K",
    "GITW_256": "GITW",
}

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


def write_metrics_csv(rows: List[Dict[str, object]], metric_csv_path: str, fieldnames: List[str] = None) -> None:
    with open(metric_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


NOT_AVAILABLE = "—"
"""Printed for a model/dataset combination that has no result file. A model evaluated on only some of
the datasets — a VLM baseline, or a run in progress — would otherwise put the literal string "nan" in
every missing cell of the published table."""


def _format_cell(value, formatter: Callable[[float], str]) -> str:
    if value is None or value != value:  # NaN: the pivot had no row for this model and dataset
        return NOT_AVAILABLE
    return formatter(value)


def _emphasise_best(column, higher_is_better: bool):
    """Bold the best entry of one already formatted column, and every entry that ties with it.

    The comparison is on the formatted value rather than the raw one, so that two cells which are printed
    identically are either both bold or neither, instead of one winning on invisible decimals. Cells with
    no result take part in neither the comparison nor the bolding.
    """
    numbers = [None if value == NOT_AVAILABLE else float(value) for value in column]
    present = [number for number in numbers if number is not None]
    if not present:
        return list(column)
    best = max(present) if higher_is_better else min(present)
    return [f"**{value}**" if number == best else value for value, number in zip(column, numbers)]


def format_results_csv_as_markdown_table(metric_csv_path: str, metrics: Dict[str, MetricSpec] = None) -> str:
    """Render one markdown table per metric, models in rows and datasets in columns."""
    import pandas as pd

    metrics = metrics if metrics is not None else HEADLINE_METRICS

    df = pd.read_csv(metric_csv_path)
    tables = []
    for metric, spec in metrics.items():
        table = df.pivot(index="model", columns="dataset", values=metric)
        table = table.rename(columns=lambda dataset: DATASET_LABELS.get(dataset, dataset))
        table = table.map(lambda value: _format_cell(value, spec.format))
        for column in table.columns:
            table[column] = _emphasise_best(table[column], spec.higher_is_better)
        # disable_numparse keeps tabulate from re-parsing the formatted strings back into numbers, which
        # would drop trailing zeroes and undo the per-metric formatting. that also loses the numeric
        # alignment, so the columns are aligned explicitly.
        alignment = ("left",) + ("right",) * len(table.columns)
        tables.append(f"**{metric}**\n\n{table.to_markdown(disable_numparse=True, colalign=alignment)}")
    return "\n\n".join(tables)


if __name__ == "__main__":
    # imported here rather than at module level: `aggregate_seed_metrics` imports this module for
    # `get_metrics`, so importing it back at the top would be a cycle.
    from kp_2d_benchmark import DATA_DIR
    from kp_2d_benchmark.eval.aggregate_seed_metrics import SEED_RESULTS_SUBDIR, aggregate_seed_rows

    results_dir = DATA_DIR / "results"
    metric_csv_path = "metrics.csv"
    spread_csv_path = "metrics_seed_spread.csv"

    rows = []
    for file in sorted(os.listdir(results_dir)):
        if file.endswith(".json"):
            file = os.path.join(results_dir, file)
            rows.append(get_metrics(file))
            print(f"metrics for {os.path.basename(file)} done")
    n_single_run_rows = len(rows)

    # models that are run once per random seed live one directory down, and contribute the mean over
    # their seeds. see kp_2d_benchmark/eval/aggregate_seed_metrics.py.
    mean_rows, spread_rows = aggregate_seed_rows(str(results_dir / SEED_RESULTS_SUBDIR))
    rows.extend(mean_rows)

    write_metrics_csv(rows, metric_csv_path)
    print(
        f"Metrics for {n_single_run_rows} result files and {len(mean_rows)} seed-averaged rows "
        f"saved to {metric_csv_path}\n"
    )
    if spread_rows:
        write_metrics_csv(spread_rows, spread_csv_path, fieldnames=["model", "dataset", "n_seeds", *METRIC_FORMATTERS])
        print(f"Standard deviation over the seeds saved to {spread_csv_path}\n")
    print(format_results_csv_as_markdown_table(metric_csv_path))
