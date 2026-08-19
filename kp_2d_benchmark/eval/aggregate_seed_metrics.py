"""Collapse the per-seed result files of a one-shot model into a single row per model and dataset.

`scripts/few_shot_matching.py` runs its configuration once per random seed, because the method it wraps
picks its single reference point per keypoint from a randomly drawn train image, and one unlucky draw can
move a whole row. Those runs land in `data/results/few-shot-seeds/` as

    model=<label>,dataset=<DatasetRepr>,seed=<seed>.json

which `calculate_all_metrics.py` never picks up on its own, since it skips subdirectories. The metrics
are computed per seed with the ordinary `get_metrics`, and only then averaged: averaging metrics is the
only thing that makes sense here, as there is nothing meaningful to average about the predictions
themselves.

Both the mean and the population standard deviation across seeds are returned. The mean is what goes into
`metrics.csv` next to the trained models; the spread goes into `metrics_seed_spread.csv`, so that a row
whose value swings wildly with the reference image cannot be read as a stable number.
"""

import os
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

from kp_2d_benchmark.eval.calculate_all_metrics import METRIC_FORMATTERS, get_metrics, parse_result_file_name

SEED_RESULTS_SUBDIR = "few-shot-seeds"


def parse_seed(file_name: str) -> int:
    """The seed of a `model=<>,dataset=<>,seed=<>.json` file name.

    The seed is the *last* field so that `parse_result_file_name` keeps working unchanged on these names.
    """
    fields = os.path.basename(file_name).split(".json")[0].split(",")
    if len(fields) != 3 or not fields[2].startswith("seed="):
        raise ValueError(f"{os.path.basename(file_name)} is not a `model=<>,dataset=<>,seed=<>.json` name")
    return int(fields[2].split("=")[1])


def group_seed_files(results_dir: str) -> Dict[Tuple[str, str], Dict[int, str]]:
    """Every per-seed result file in `results_dir`, grouped by (model, dataset) and keyed by seed."""
    groups: Dict[Tuple[str, str], Dict[int, str]] = defaultdict(dict)
    for file_name in sorted(os.listdir(results_dir)):
        if not file_name.endswith(".json"):
            continue
        model, dataset_name = parse_result_file_name(file_name)
        seed = parse_seed(file_name)
        groups[(model, dataset_name)][seed] = os.path.join(results_dir, file_name)
    return groups


def check_seed_sets_agree(groups: Dict[Tuple[str, str], Dict[int, str]]) -> None:
    """Raise unless every group of one model was run at exactly the same seeds.

    A dataset that is still running, or one whose last seed crashed, would otherwise contribute a row
    averaged over fewer draws than its neighbours, and nothing in the published table would say so.
    """
    seeds_per_model: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
    for (model, dataset_name), files in groups.items():
        seeds_per_model[model][dataset_name] = sorted(files)

    for model, seeds_per_dataset in seeds_per_model.items():
        distinct = {tuple(seeds) for seeds in seeds_per_dataset.values()}
        if len(distinct) > 1:
            raise ValueError(
                f"{model} was not run at the same seeds on every dataset, so its rows would be averaged "
                f"over different numbers of draws: "
                + ", ".join(f"{dataset}={seeds}" for dataset, seeds in sorted(seeds_per_dataset.items()))
            )


def aggregate_seed_rows(results_dir: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """(mean rows, standard deviation rows) over the seeds of every model and dataset in `results_dir`.

    The `n_seeds` of each group is carried on the spread row, which is where a reader goes to ask how
    trustworthy the mean is.
    """
    if not os.path.isdir(results_dir):
        return [], []

    groups = group_seed_files(results_dir)
    check_seed_sets_agree(groups)

    mean_rows, spread_rows = [], []
    for (model, dataset_name), files in sorted(groups.items()):
        per_seed = [get_metrics(files[seed]) for seed in sorted(files)]
        print(f"metrics for {model} on {dataset_name} over {len(per_seed)} seeds done")

        mean_row: Dict[str, object] = {"model": model, "dataset": dataset_name}
        spread_row: Dict[str, object] = {"model": model, "dataset": dataset_name, "n_seeds": len(per_seed)}
        for metric in METRIC_FORMATTERS:
            values = [row[metric] for row in per_seed]
            mean_row[metric] = statistics.fmean(values)
            # population rather than sample standard deviation: these five seeds are the whole set of
            # draws that produced the mean, not a sample of a larger population of runs.
            spread_row[metric] = statistics.pstdev(values)
        mean_rows.append(mean_row)
        spread_rows.append(spread_row)

    return mean_rows, spread_rows


if __name__ == "__main__":
    from kp_2d_benchmark import DATA_DIR

    means, spreads = aggregate_seed_rows(str(DATA_DIR / "results" / SEED_RESULTS_SUBDIR))
    for row in means:
        print(row)
    for row in spreads:
        print(row)
