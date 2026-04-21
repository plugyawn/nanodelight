from __future__ import annotations

import math
from typing import Literal

MetricDirection = Literal["maximize", "minimize", "diagnostic"]

MAXIMIZE_EXACT = {
    "validation/accuracy",
    "train/reward",
    "train/reward_mean",
    "train/natural_termination_rate",
    "train/kondo_selected_token_recall",
}

MINIMIZE_EXACT = {
    "train/loss",
    "train/truncation_rate",
    "train/gen_kl_error",
    "train/policy_kl_error",
    "train/kl_penalty",
    "train/kondo_bypass_step",
    "timing/setup/total_setup_time_s",
    "timing/train/total_step_time",
}

DIAGNOSTIC_PREFIXES = (
    "train/dg_",
    "train/kondo_",
    "train/advantages/",
)

MAXIMIZE_SUBSTRINGS = (
    "accuracy",
    "tokens_per_sec",
    "samples_per_sec",
    "throughput",
    "utilization",
    "mfu",
    "flops",
    "acceptance_rate",
)

MINIMIZE_SUBSTRINGS = (
    "loss",
    "error",
    "truncation",
    "kl",
    "latency",
)


def metric_direction(metric: str) -> MetricDirection:
    lower_metric = metric.lower()
    if metric in MAXIMIZE_EXACT:
        return "maximize"
    if metric in MINIMIZE_EXACT:
        return "minimize"
    if metric.startswith(DIAGNOSTIC_PREFIXES) or lower_metric.endswith("_std"):
        return "diagnostic"
    if lower_metric.startswith("timing/"):
        if any(token in lower_metric for token in MAXIMIZE_SUBSTRINGS):
            return "maximize"
        return "minimize"
    if any(token in lower_metric for token in MINIMIZE_SUBSTRINGS):
        return "minimize"
    if any(token in lower_metric for token in MAXIMIZE_SUBSTRINGS):
        return "maximize"
    return "diagnostic"


def finite_points(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    filtered: list[tuple[int, float]] = []
    for step, value in points:
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            filtered.append((int(step), numeric_value))
    return filtered


def summarize_metric_points(
    points: list[tuple[int, float]],
    metric_name: str,
) -> dict[str, float | int | str | None] | None:
    if not points:
        return None

    direction = metric_direction(metric_name)
    total_points = [(int(step), float(value)) for step, value in points]
    final_step, final_value = total_points[-1]
    final_is_finite = math.isfinite(final_value)
    finite = finite_points(points)

    min_step: int | None = None
    min_value: float | None = None
    max_step: int | None = None
    max_value: float | None = None
    if finite:
        min_step, min_value = min(finite, key=lambda item: item[1])
        max_step, max_value = max(finite, key=lambda item: item[1])

    summary: dict[str, float | int | str | None] = {
        "direction": direction,
        "num_points": len(total_points),
        "num_finite_points": len(finite),
        "final_step": final_step,
        "final_value": final_value,
        "final_is_finite": final_is_finite,
        "min_step": min_step,
        "min_value": min_value,
        "max_step": max_step,
        "max_value": max_value,
        "best_step": None,
        "best_value": None,
        "worst_step": None,
        "worst_value": None,
    }

    if direction == "maximize" and max_step is not None and min_step is not None:
        summary["best_step"] = max_step
        summary["best_value"] = max_value
        summary["worst_step"] = min_step
        summary["worst_value"] = min_value
    elif direction == "minimize" and max_step is not None and min_step is not None:
        summary["best_step"] = min_step
        summary["best_value"] = min_value
        summary["worst_step"] = max_step
        summary["worst_value"] = max_value

    return summary


def is_gate_neutral(gate_mean: float | None, gate_spread: float | None) -> bool:
    if gate_mean is None or gate_spread is None:
        return False
    if not math.isfinite(gate_mean) or not math.isfinite(gate_spread):
        return False
    return 0.45 <= gate_mean <= 0.55 and abs(gate_spread) < 0.05
