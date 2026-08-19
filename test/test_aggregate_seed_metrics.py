"""Tests for collapsing per-seed result files into one row.

The metrics themselves are covered by `test_metrics.py`; what matters here is that the right files end up
in the right group, that the mean and spread are taken over all of them, and that a group which is missing
a seed is refused rather than silently averaged over fewer draws.
"""

import json
import os

import pytest

from kp_2d_benchmark.eval.aggregate_seed_metrics import (
    aggregate_seed_rows,
    check_seed_sets_agree,
    group_seed_files,
    parse_seed,
)

MODEL = "fsk-dinov3-s"
DATASET = "GITW_256"
SEEDS = [2025, 2026, 2027]


def _file_name(model=MODEL, dataset=DATASET, seed=2025):
    return f"model={model},dataset={dataset},seed={seed}.json"


def _write_seed_files(directory, models=(MODEL,), datasets=(DATASET,), seeds=SEEDS):
    for model in models:
        for dataset in datasets:
            for seed in seeds:
                path = os.path.join(directory, _file_name(model, dataset, seed))
                with open(path, "w") as f:
                    json.dump([], f)


def test_parse_seed():
    assert parse_seed(_file_name(seed=2029)) == 2029
    assert parse_seed(os.path.join("a", "b", _file_name(seed=7))) == 7


@pytest.mark.parametrize(
    "file_name",
    [
        "model=fsk-dinov3-s,dataset=GITW_256.json",  # the shape the other models write
        "model=fsk-dinov3-s,seed=2025,dataset=GITW_256.json",  # seed must be last, so that
        # parse_result_file_name keeps reading the dataset out of the second field
        "fsk-dinov3-s-2025.json",
    ],
)
def test_parse_seed_rejects_other_names(file_name):
    with pytest.raises(ValueError):
        parse_seed(file_name)


def test_group_seed_files_groups_by_model_and_dataset(tmp_path):
    _write_seed_files(tmp_path, models=("a", "b"), datasets=("X", "Y"))
    groups = group_seed_files(str(tmp_path))

    assert set(groups) == {("a", "X"), ("a", "Y"), ("b", "X"), ("b", "Y")}
    for files in groups.values():
        assert sorted(files) == SEEDS


def test_group_seed_files_ignores_non_json(tmp_path):
    _write_seed_files(tmp_path)
    (tmp_path / "notes.txt").write_text("not a result file")
    assert len(group_seed_files(str(tmp_path))) == 1


def test_check_seed_sets_agree_accepts_matching_seeds(tmp_path):
    _write_seed_files(tmp_path, datasets=("X", "Y"))
    check_seed_sets_agree(group_seed_files(str(tmp_path)))


def test_check_seed_sets_agree_rejects_a_missing_seed(tmp_path):
    _write_seed_files(tmp_path, datasets=("X",), seeds=SEEDS)
    _write_seed_files(tmp_path, datasets=("Y",), seeds=SEEDS[:-1])

    with pytest.raises(ValueError, match="not be run at the same seeds|not run at the same seeds"):
        check_seed_sets_agree(group_seed_files(str(tmp_path)))


def test_check_seed_sets_agree_allows_two_models_at_different_seeds(tmp_path):
    """Seeds have to agree within a model, not between models: a second model may be run separately."""
    _write_seed_files(tmp_path, models=("a",), seeds=[1, 2])
    _write_seed_files(tmp_path, models=("b",), seeds=[3, 4, 5])
    check_seed_sets_agree(group_seed_files(str(tmp_path)))


def test_aggregate_seed_rows_returns_nothing_for_a_missing_directory(tmp_path):
    assert aggregate_seed_rows(str(tmp_path / "does-not-exist")) == ([], [])


def test_aggregate_seed_rows_averages_the_metrics(tmp_path, monkeypatch):
    """The mean is over the *metrics* of each seed, and the spread is the population standard deviation."""
    _write_seed_files(tmp_path, seeds=[2025, 2026])
    per_file_median_nme = {2025: 0.10, 2026: 0.20}

    def fake_get_metrics(file_path):
        return {
            "model": MODEL,
            "dataset": DATASET,
            "detection_rate": 1.0,
            "median_nme": per_file_median_nme[parse_seed(file_path)],
            "pck@0.05": 0.5,
            "strict_success@0.05": 0.25,
            "mAP@0.05": 0.4,
            "average_keypoint_distance": 10.0,
            "median_keypoint_distance": 8.0,
        }

    monkeypatch.setattr("kp_2d_benchmark.eval.aggregate_seed_metrics.get_metrics", fake_get_metrics)
    (means,), (spreads,) = aggregate_seed_rows(str(tmp_path))

    assert means["model"] == MODEL and means["dataset"] == DATASET
    assert means["median_nme"] == pytest.approx(0.15)
    assert means["detection_rate"] == pytest.approx(1.0)

    assert spreads["n_seeds"] == 2
    assert spreads["median_nme"] == pytest.approx(0.05)  # population sigma of {0.10, 0.20}
    assert spreads["detection_rate"] == pytest.approx(0.0)


def test_aggregated_row_has_the_metrics_csv_columns(tmp_path, monkeypatch):
    """A mean row is written straight into metrics.csv, so it has to carry exactly those fields."""
    from kp_2d_benchmark.eval.calculate_all_metrics import CSV_FIELDS

    _write_seed_files(tmp_path, seeds=[2025])
    monkeypatch.setattr(
        "kp_2d_benchmark.eval.aggregate_seed_metrics.get_metrics",
        lambda file_path: {field: 1.0 for field in CSV_FIELDS} | {"model": MODEL, "dataset": DATASET},
    )
    (means,), _ = aggregate_seed_rows(str(tmp_path))
    assert set(means) == set(CSV_FIELDS)
