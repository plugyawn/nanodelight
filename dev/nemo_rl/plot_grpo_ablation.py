#!/usr/bin/env python3
"""Plot scalar GRPO ablations from TensorBoard and JSONL artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

try:
    from .metric_semantics import finite_points, summarize_metric_points
except ImportError:
    from metric_semantics import finite_points, summarize_metric_points

SIZE_GUIDANCE = {
    event_accumulator.TENSORS: 0,
    event_accumulator.SCALARS: 0,
}

DEFAULT_METRICS = [
    "validation/accuracy",
    "train/loss",
    "train/advantages/mean",
    "train/dg_gate_mean",
    "train/dg_gate_positive_mean",
    "train/dg_gate_negative_mean",
    "train/dg_gate_spread",
    "train/dg_surprisal_mean",
    "train/kondo_actual_backward_token_fraction",
    "train/kondo_rows_kept_fraction",
]

KONDO_JSONL_OVERRIDE_METRICS = {
    "train/kondo_actual_backward_token_fraction",
    "train/kondo_skipped_token_fraction",
    "train/kondo_rows_kept_fraction",
    "train/kondo_rows_kept",
    "train/kondo_rows_total",
}


@dataclass
class RunData:
    label: str
    run_dir: str
    metrics: dict[str, list[tuple[int, float]]]


def summarize_metrics(
    metrics: dict[str, list[tuple[int, float]]],
) -> dict[str, dict[str, float | int | str | None]]:
    summary: dict[str, dict[str, float | int | str | None]] = {}
    for name, points in metrics.items():
        metric_summary = summarize_metric_points(points, name)
        if metric_summary is None:
            continue
        summary[name] = metric_summary
    return summary


def load_tensorboard_scalars(run_dir: str) -> dict[str, dict[int, float]]:
    event_files = sorted(
        glob.glob(
            os.path.join(run_dir, "**", "tensorboard", "**", "events*tfevents*"),
            recursive=True,
        )
    )
    merged: dict[str, dict[int, float]] = {}

    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            event_file, size_guidance=SIZE_GUIDANCE
        )
        accumulator.Reload()
        for metric_name in accumulator.scalars.Keys():
            metric_steps = merged.setdefault(metric_name, {})
            for scalar in accumulator.Scalars(metric_name):
                metric_steps[scalar.step] = scalar.value

    return merged


def load_reward_stats(run_dir: str) -> dict[str, dict[int, float]]:
    step_files = sorted(
        glob.glob(os.path.join(run_dir, "**", "train_data_step*.jsonl"), recursive=True)
    )
    reward_mean: dict[int, float] = {}
    reward_std: dict[int, float] = {}

    for path in step_files:
        step_str = os.path.splitext(os.path.basename(path))[0].replace(
            "train_data_step", ""
        )
        step = int(step_str)
        rewards: list[float] = []
        with open(path) as handle:
            for line in handle:
                sample = json.loads(line)
                value = sample.get("rewards")
                if isinstance(value, list):
                    rewards.extend(float(x) for x in value)
                elif value is not None:
                    rewards.append(float(value))
        if rewards:
            mean = sum(rewards) / len(rewards)
            var = sum((x - mean) ** 2 for x in rewards) / len(rewards)
            reward_mean[step] = mean
            reward_std[step] = var**0.5

    return {
        "train/reward_mean": reward_mean,
        "train/reward_std": reward_std,
    }


def _flatten_numeric(value: object) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    return []


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def load_jsonl_metrics(run_dir: str) -> dict[str, dict[int, float]]:
    metric_map: dict[str, dict[int, float]] = {}

    train_files = sorted(
        glob.glob(os.path.join(run_dir, "**", "train_data_step*.jsonl"), recursive=True)
    )
    for path in train_files:
        step = int(
            os.path.splitext(os.path.basename(path))[0].replace("train_data_step", "")
        )
        rewards: list[float] = []
        total_valid_tokens = 0.0
        total_backward_tokens = 0.0
        total_rows = 0.0
        kept_rows = 0.0
        with open(path) as handle:
            for line in handle:
                sample = json.loads(line)
                reward_values = _flatten_numeric(sample.get("rewards"))
                rewards.extend(reward_values)

                token_mask = _flatten_numeric(sample.get("token_loss_mask"))
                valid_tokens = sum(token_mask)
                total_valid_tokens += valid_tokens
                total_rows += 1.0

                row_selected_values = _flatten_numeric(sample.get("kondo_row_selected"))
                row_selected = bool(row_selected_values and row_selected_values[0] >= 0.5)
                if row_selected or "kondo_row_selected" not in sample:
                    kept_rows += 1.0

                if "loss_token_mask" in sample:
                    total_backward_tokens += sum(
                        _flatten_numeric(sample.get("loss_token_mask"))
                    )
                elif "kondo_row_selected" in sample:
                    if row_selected:
                        total_backward_tokens += valid_tokens
                else:
                    total_backward_tokens += valid_tokens

        reward_mean = _mean(rewards)
        if reward_mean is not None:
            metric_map.setdefault("train/reward_mean", {})[step] = reward_mean
        if rewards:
            var = sum((x - reward_mean) ** 2 for x in rewards) / len(rewards)
            metric_map.setdefault("train/reward_std", {})[step] = var**0.5
        if total_valid_tokens > 0:
            fraction = total_backward_tokens / total_valid_tokens
            metric_map.setdefault(
                "train/kondo_actual_backward_token_fraction", {}
            )[step] = fraction
            metric_map.setdefault("train/kondo_skipped_token_fraction", {})[
                step
            ] = 1.0 - fraction
        if total_rows > 0:
            metric_map.setdefault("train/kondo_rows_kept_fraction", {})[step] = (
                kept_rows / total_rows
            )
            metric_map.setdefault("train/kondo_rows_kept", {})[step] = kept_rows
            metric_map.setdefault("train/kondo_rows_total", {})[step] = total_rows

    val_files = sorted(
        glob.glob(os.path.join(run_dir, "**", "val_data_step*.jsonl"), recursive=True)
    )
    for path in val_files:
        step = int(
            os.path.splitext(os.path.basename(path))[0].replace("val_data_step", "")
        )
        rewards: list[float] = []
        with open(path) as handle:
            for line in handle:
                sample = json.loads(line)
                rewards.extend(_flatten_numeric(sample.get("rewards")))
        accuracy = _mean(rewards)
        if accuracy is not None:
            metric_map.setdefault("validation/accuracy", {})[step] = accuracy

    return metric_map


def merge_metric_maps(
    tensorboard_metrics: dict[str, dict[int, float]],
    jsonl_metrics: dict[str, dict[int, float]],
    reward_metrics: dict[str, dict[int, float]],
) -> dict[str, dict[int, float]]:
    metric_map = {
        name: dict(points) for name, points in tensorboard_metrics.items()
    }
    for name, points in jsonl_metrics.items():
        metric_steps = metric_map.setdefault(name, {})
        for step, value in points.items():
            if name in KONDO_JSONL_OVERRIDE_METRICS:
                metric_steps[step] = value
            else:
                metric_steps.setdefault(step, value)
    for name, points in reward_metrics.items():
        metric_steps = metric_map.setdefault(name, {})
        for step, value in points.items():
            metric_steps.setdefault(step, value)
    return metric_map


def normalize_metrics(
    metric_map: dict[str, dict[int, float]],
) -> dict[str, list[tuple[int, float]]]:
    return {
        name: sorted((int(step), float(value)) for step, value in points.items())
        for name, points in metric_map.items()
    }


def parse_run_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=RUN_DIR")
    label, run_dir = value.split("=", 1)
    return label, run_dir


def plot_metric(metric_name: str, runs: list[RunData], out_dir: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plotted = False
    for run in runs:
        raw_points = run.metrics.get(metric_name, [])
        if not raw_points:
            continue
        finite = finite_points(raw_points)
        if not finite:
            continue
        steps = [int(step) for step, _ in raw_points]
        values = [
            float(value) if math.isfinite(float(value)) else float("nan")
            for _, value in raw_points
        ]
        (line,) = plt.plot(
            steps,
            values,
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=run.label,
        )
        nonfinite_steps = [
            int(step) for step, value in raw_points if not math.isfinite(float(value))
        ]
        if nonfinite_steps:
            marker_y = max(value for _, value in finite)
            plt.scatter(
                nonfinite_steps,
                [marker_y] * len(nonfinite_steps),
                marker="x",
                s=28,
                color=line.get_color(),
            )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(metric_name)
    plt.xlabel("step")
    plt.ylabel(metric_name.split("/")[-1])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    filename = metric_name.replace("/", "__") + ".png"
    plt.savefig(os.path.join(out_dir, filename), dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run specification as LABEL=RUN_DIR",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric to plot. Can be passed multiple times.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for plots and merged JSON")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_to_plot = args.metric or DEFAULT_METRICS
    runs: list[RunData] = []
    merged_output: dict[str, dict[str, list[tuple[int, float]]]] = {}
    summary_output: dict[str, dict[str, dict[str, float | int | str | None]]] = {}

    for label, run_dir in args.run:
        metric_map = merge_metric_maps(
            tensorboard_metrics=load_tensorboard_scalars(run_dir),
            jsonl_metrics=load_jsonl_metrics(run_dir),
            reward_metrics=load_reward_stats(run_dir),
        )
        normalized = normalize_metrics(metric_map)
        runs.append(RunData(label=label, run_dir=run_dir, metrics=normalized))
        merged_output[label] = normalized
        summary_output[label] = summarize_metrics(normalized)

    with open(os.path.join(args.out_dir, "merged_metrics.json"), "w") as handle:
        json.dump(merged_output, handle, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as handle:
        json.dump(summary_output, handle, indent=2)
    with open(os.path.join(args.out_dir, "summary.tsv"), "w") as handle:
        handle.write(
            "run\tmetric\tdirection\tnum_points\tnum_finite_points\tfinal_step\tfinal_value\tfinal_is_finite\tmin_step\tmin_value\tmax_step\tmax_value\tbest_step\tbest_value\tworst_step\tworst_value\n"
        )
        for label, metric_summary in summary_output.items():
            for metric_name, stats in sorted(metric_summary.items()):
                handle.write(
                    "\t".join(
                        [
                            label,
                            metric_name,
                            str(stats["direction"]),
                            str(stats["num_points"]),
                            str(stats["num_finite_points"]),
                            str(stats["final_step"]),
                            str(stats["final_value"]),
                            str(stats["final_is_finite"]),
                            str(stats["min_step"]),
                            str(stats["min_value"]),
                            str(stats["max_step"]),
                            str(stats["max_value"]),
                            "" if stats["best_step"] is None else str(stats["best_step"]),
                            "" if stats["best_value"] is None else str(stats["best_value"]),
                            "" if stats["worst_step"] is None else str(stats["worst_step"]),
                            "" if stats["worst_value"] is None else str(stats["worst_value"]),
                        ]
                    )
                    + "\n"
                )

    for metric_name in metrics_to_plot:
        plot_metric(metric_name, runs, args.out_dir)


if __name__ == "__main__":
    main()
