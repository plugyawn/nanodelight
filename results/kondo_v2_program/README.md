# Kondo V2 Candidate Program

This directory tracks the bounded v2 research program for delight-driven compute skip in
NeMo-RL. The goal is to converge on a production-real, testable Kondo form rather than
continue exploring loosely connected variants.

## Shared Standards

- Reference baseline: plain GRPO
- Dense teacher: exact dense top-k delight token mask on the learner batch
- Primary quality metric: held-out `gsm8k:test` accuracy
- Primary fidelity metric: `kondo_selected_token_recall`
- Primary compute metric: `kondo_kept_row_token_fraction`
- Primary runtime metric: `timing/train/policy_training`
- Candidate win condition:
  - beats or matches baseline quality across matched seeds
  - preserves dense-support fidelity within a declared floor
  - produces real policy-training speedup

## Candidate Threads

### Thread A: Row-Cover Kondo V2

Status: active

Definition:
- exact dense teacher
- response-level row cover
- exact dense `loss_token_mask` inside kept rows
- full-batch `loss_normalizer` and `kl_normalizer`
- explicit recall floor

Purpose:
- establish the strongest row-executed production-real candidate

### Thread B: Block-Cover Executor Feasibility

Status: frozen pending a new executor

Definition:
- exact dense teacher
- block cover as an execution approximation
- only considered viable if it buys real compute beyond row compaction

Purpose:
- determine whether current block variants are worth continuing without a new executor

### Thread C: Async-Safe Kondo

Status: active

Definition:
- freshness-aware async gating
- dense-teacher telemetry always on
- structured routing only if freshness and estimator concerns are satisfied

Purpose:
- converge on the first async-safe Kondo form for NeMo-RL

### Thread D: Measurement and Claim Integrity

Status: active

Definition:
- bundle consistency
- launcher/config consistency
- optimizer-side compute counters
- documentation and tests

Purpose:
- ensure every candidate is judged on valid, reproducible evidence

## Candidate Exit Criteria

A thread is only considered complete when:

1. The implementation exists locally with tests.
2. The A100 ablations for that thread have run to a stable conclusion.
3. A redteam pass has checked for meaningful remaining optimizations.
4. The conclusion is recorded here with surviving follow-ups or a kill decision.

## Current Working Hypothesis

The current best v1/v2 bridge is:

`exact dense teacher -> recall-constrained row cover -> packed-row executor`

This is effectively a cleaned-up `response_dense_rows` with stricter measurement and
fidelity contracts. Current block variants are treated as provisional until they show
real execution benefits beyond row compaction.

## Current Verdicts

- Surviving sync candidate:
  - `response_dense_rows_v2`
  - exact dense teacher support
  - recall-constrained row cover
  - packed-row execution only
- Frozen threads:
  - current row-executed block family
  - legacy `response_routed` / denominator-tuning variants
- Async first trusted form:
  - prompt-group-preserving dense-row routing with freshness guards
  - no stale-group competition
- External v2 ideas worth revisiting only after v1 is clean:
  - reversible middle-layer token dropping with restore
  - Random-LTD as a control, not the main path
  - MoD-style routed blocks only with a new executor

## Active Batch

- Manifest:
  - [gsm8k_v2_minimal.json](/Users/progyan/nanodelight/results/kondo_v2_program/manifests/gsm8k_v2_minimal.json)
- Remote run root:
  - `/root/nanodelight-run/kondo_v2_family_20260421T163443Z`
- Status:
  - live on the Prime A100
  - matrix = `2 seeds x 4 arms`
