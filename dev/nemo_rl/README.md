# NeMo-RL DG-GRPO Reference Path

This directory holds the downstream integration we used to prototype Delightful GRPO on a research-grade RL stack before touching nanochat's smaller `chat_rl.py` loop.

What lives here:

- `patches/0001-dg-grpo-loss-and-metrics.patch`
  Adds DG gate support to NeMo-RL's clipped policy-gradient loss, exposes `dg_enabled` / `dg_eta`, and fixes GRPO metric aggregation so DG diagnostics are logged correctly.
- `patches/0002-dtensor-tokenizer-bootstrap.patch`
  Fixes DTensor v1 worker startup by reconstructing the tokenizer inside the worker instead of pickling it across env boundaries.
- `plot_grpo_ablation.py`
  Merges TensorBoard scalars and per-step JSONL rewards into plots and compact JSON summaries.
- `analyze_grpo_rollouts.py`
  Produces a stability-focused summary across one or more GRPO runs.
- `diag_grpo_setup.py`
  Lightweight setup smoke test for NeMo-RL GRPO configs.

The companion launcher in [runs/nemo_grpo_gsm8k_full.sh](../../runs/nemo_grpo_gsm8k_full.sh) assumes you have a separate NeMo-RL checkout available via `NEMO_RL_ROOT`.

## Apply The Patches

```bash
cd /path/to/NeMo-RL
git apply /path/to/nanodelight/dev/nemo_rl/patches/0001-dg-grpo-loss-and-metrics.patch
git apply /path/to/nanodelight/dev/nemo_rl/patches/0002-dtensor-tokenizer-bootstrap.patch
```

The launcher can also do this automatically when `APPLY_PATCHES=1`.

## Kick Off A Run

Baseline:

```bash
NEMO_RL_ROOT=/path/to/NeMo-RL \
DG_ENABLED=0 \
bash runs/nemo_grpo_gsm8k_full.sh
```

DG:

```bash
NEMO_RL_ROOT=/path/to/NeMo-RL \
DG_ENABLED=1 \
DG_ETA=0.5 \
RUN_NAME=gsm8k_dg_eta0p5_seed42 \
bash runs/nemo_grpo_gsm8k_full.sh
```

## Plot And Analyze

```bash
python dev/nemo_rl/plot_grpo_ablation.py \
  --run baseline=/path/to/baseline_run \
  --run dg_eta0p5=/path/to/dg_run \
  --metric validation/accuracy \
  --metric train/reward \
  --metric train/dg_gate_mean \
  --out-dir /tmp/dg_plots
```

```bash
python dev/nemo_rl/analyze_grpo_rollouts.py \
  --run baseline=/path/to/baseline_run \
  --run dg_eta0p5=/path/to/dg_run \
  --out-dir /tmp/dg_analysis
```

## Notes From The First Screen

- `eta=1.0` underperformed the baseline on the 8-step GSM8K screen.
- `eta=0.5` matched the best short-horizon validation accuracy from the screen and is the only DG setting worth extending first.
- All three runs showed high truncation rates with `max_new_tokens=256`, so longer-horizon studies should watch truncation carefully.
- DG gate means stayed near `0.5` with zero spread in the 8-step screen, which is stable but suggests the DG signal is still weak in that regime.
