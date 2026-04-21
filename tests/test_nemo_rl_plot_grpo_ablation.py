from dev.nemo_rl.plot_grpo_ablation import merge_metric_maps


def test_merge_metric_maps_prefers_jsonl_for_kondo_compute_metrics():
    tensorboard_metrics = {
        "train/kondo_actual_backward_token_fraction": {8: 0.61},
        "train/kondo_rows_kept_fraction": {8: 0.75},
        "validation/accuracy": {8: 0.45},
    }
    jsonl_metrics = {
        "train/kondo_actual_backward_token_fraction": {8: 0.70},
        "train/kondo_rows_kept_fraction": {8: 0.80},
        "validation/accuracy": {8: 0.46},
    }
    reward_metrics = {
        "train/reward_mean": {8: 0.25},
    }

    merged = merge_metric_maps(
        tensorboard_metrics=tensorboard_metrics,
        jsonl_metrics=jsonl_metrics,
        reward_metrics=reward_metrics,
    )

    assert merged["train/kondo_actual_backward_token_fraction"][8] == 0.70
    assert merged["train/kondo_rows_kept_fraction"][8] == 0.80
    assert merged["validation/accuracy"][8] == 0.45
    assert merged["train/reward_mean"][8] == 0.25
