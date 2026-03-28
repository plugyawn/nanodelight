# GSM8K DG Screen (2026-03-28)

This directory contains the first short-horizon DG-vs-GRPO screen we ran on top of patched NeMo-RL.

## Setup

- Stack: NeMo-RL plus the downstream patches in [dev/nemo_rl/patches](../../dev/nemo_rl/patches)
- Model: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- Task: built-in `gsm8k` train/test recipe
- Seed: `42`
- Horizon: `8` GRPO steps
- Batch shape: `4` prompts per step x `8` generations per prompt = `32` sampled responses per update
- Validation: `64` test samples every `4` steps
- Generation cap: `max_new_tokens=256`

This is a screen, not a claim-quality study. Across `8` steps we only touch about `32` unique GSM8K prompts.

## Outcome

Short-horizon validation favored `dg_eta0p5`.

| run | final val acc | best val acc | final reward | best reward |
|---|---:|---:|---:|---:|
| baseline | 0.34375 | 0.37500 | 0.50000 | 0.78125 |
| dg_eta1p0 | 0.31250 | 0.328125 | 0.50000 | 0.84375 |
| dg_eta0p5 | 0.37500 | 0.37500 | 0.40625 | 0.81250 |

Interpretation:

- `eta=1.0` is not worth extending first. It trailed the baseline on validation accuracy.
- `eta=0.5` is the only DG setting that held up on validation in this short screen.
- The DG runs were not obviously unstable, but the DG signal also looked weak.

## Stability Probes

- All three runs completed without crashes, NaNs, or exploding losses.
- Final `train/gen_kl_error` stayed tiny:
  - baseline: `1.92e-4`
  - `dg_eta1p0`: `1.52e-4`
  - `dg_eta0p5`: `1.69e-4`
- `train/natural_termination_rate` stayed at `1.0` for all runs.
- Truncation stayed high at `max_new_tokens=256`:
  - baseline final `train/truncation_rate`: `0.71875`
  - `dg_eta1p0` final `train/truncation_rate`: `0.78125`
  - `dg_eta0p5` final `train/truncation_rate`: `0.68750`
- DG gate diagnostics were nearly neutral:
  - `dg_eta1p0` final `train/dg_gate_mean`: `0.49826`
  - `dg_eta0p5` final `train/dg_gate_mean`: `0.50047`
  - both DG runs had final `train/dg_gate_spread = 0.0`

The zero gate spread is the main caveat from this screen. Either the DG signal is genuinely weak in this tiny regime, or the logging / masking path still is not exposing the asymmetry we want to see.

## Runtime Notes

- Setup dominated each run:
  - baseline `timing/setup/total_setup_time_s`: `622.21`
  - `dg_eta1p0`: `612.56`
  - `dg_eta0p5`: `613.99`
- Final per-step train time landed between `47s` and `54s`.
- Because startup cost is so large, very short NeMo-RL runs are mostly screening passes.

## Files

- `dg_gsm8k_compare_seed42/metric_snapshot.json`
  Selected scalar traces for the three runs.
- `dg_gsm8k_compare_seed42/metric_summary.json`
  Final/best summary for the selected metrics.
- `dg_gsm8k_compare_seed42/plots/validation__accuracy.png`
  Validation accuracy comparison.
- `dg_gsm8k_compare_seed42/plots/train__reward.png`
  Training reward comparison.
- `dg_gsm8k_compare_seed42/plots/train__dg_gate_mean.png`
  DG gate mean over time.
- `dg_gsm8k_compare_seed42/plots/train__dg_gate_spread.png`
  DG gate spread over time.
- `dg_gsm8k_compare_seed42/plots/timing__train__total_step_time.png`
  Per-step wall-clock comparison.
