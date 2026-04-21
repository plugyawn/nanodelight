from dev.nemo_rl.metric_semantics import (
    is_gate_neutral,
    metric_direction,
    summarize_metric_points,
)


def test_metric_direction_classification():
    assert metric_direction("validation/accuracy") == "maximize"
    assert metric_direction("train/kondo_selected_token_recall") == "maximize"
    assert metric_direction("train/kondo_bypass_step") == "minimize"
    assert metric_direction("train/kondo_actual_backward_token_fraction") == "diagnostic"
    assert metric_direction("train/loss") == "minimize"
    assert metric_direction("train/truncation_rate") == "minimize"
    assert metric_direction("train/gen_kl_error") == "minimize"
    assert metric_direction("timing/train/total_step_time") == "minimize"
    assert metric_direction("timing/train/valid_tokens_per_sec_per_gpu") == "maximize"
    assert metric_direction("train/dg_gate_spread") == "diagnostic"


def test_summarize_metric_points_respects_minimize_and_filters_nonfinite():
    summary = summarize_metric_points(
        [(1, 3.0), (2, float("nan")), (3, 1.0), (4, 2.0)],
        "train/loss",
    )

    assert summary is not None
    assert summary["direction"] == "minimize"
    assert summary["num_points"] == 4
    assert summary["num_finite_points"] == 3
    assert summary["final_step"] == 4
    assert summary["final_value"] == 2.0
    assert summary["best_step"] == 3
    assert summary["best_value"] == 1.0
    assert summary["worst_step"] == 1
    assert summary["worst_value"] == 3.0


def test_summarize_metric_points_preserves_terminal_nonfinite_value():
    summary = summarize_metric_points(
        [(1, 1.0), (2, 0.5), (3, float("inf"))],
        "train/loss",
    )

    assert summary is not None
    assert summary["final_step"] == 3
    assert summary["final_value"] == float("inf")
    assert summary["final_is_finite"] is False
    assert summary["num_points"] == 3
    assert summary["num_finite_points"] == 2
    assert summary["best_step"] == 2
    assert summary["best_value"] == 0.5


def test_summarize_metric_points_treats_dg_metrics_as_diagnostic():
    summary = summarize_metric_points(
        [(1, 0.4), (2, float("inf")), (3, 0.6)],
        "train/dg_gate_spread",
    )

    assert summary is not None
    assert summary["direction"] == "diagnostic"
    assert summary["min_value"] == 0.4
    assert summary["max_value"] == 0.6
    assert summary["best_step"] is None
    assert summary["worst_step"] is None


def test_gate_neutral_requires_finite_gate_metrics():
    assert is_gate_neutral(0.5, 0.0)
    assert not is_gate_neutral(None, 0.0)
    assert not is_gate_neutral(0.5, None)
    assert not is_gate_neutral(float("nan"), 0.0)
