# Kondo V2 Thread Registry

## Active Threads

### Thread A: Row-Cover Kondo V2

- Agent: `019db0d7-395a-7ae1-839f-f62531e80ee3`
- Nickname: `Laplace`
- Goal:
  - define the exact `response_dense_rows_v2` contract
  - include recall-floor logic, metrics, tests, and ablations
- Completion bar:
  - concrete implementation contract
  - ablation matrix
  - kill-list for dominated variants
- Current verdict:
  - ship `response_dense_rows_v2`
  - exact dense teacher inside kept rows
  - recall-constrained row cover
  - kill legacy `routed`, `response_0p7`, and block families on the current executor

### Thread B: Block-Cover Executor Feasibility

- Agent: `019db0d7-45e4-7b31-84f4-15e5f863ec87`
- Nickname: `Kepler`
- Goal:
  - determine whether current row-executed block variants are still worth iteration
- Completion bar:
  - one last non-redundant experiment, or a kill decision with evidence
- Current verdict:
  - kill/freeze until a real block or suffix executor exists

### Thread C: Async-Safe Kondo

- Agent: `019db0d7-5203-7461-8eb5-81bebfb1abe1`
- Nickname: `Cicero`
- Goal:
  - define the first async-safe Kondo form for NeMo-RL
- Completion bar:
  - trusted minimal design
  - freshness rules
  - test plan
  - anti-pattern list
- Current verdict:
  - first trusted async design is prompt-group-preserving dense-row routing with freshness guards

### Thread D: Measurement and Claim Integrity

- Agent: `019db0d7-67b9-7113-94a6-8ad50bcba952`
- Nickname: `Avicenna`
- Goal:
  - identify all bundle / launcher / metric inconsistencies that would invalidate a claim run
- Completion bar:
  - prioritized fix list
  - reproducibility checklist
- Current verdict:
  - freeze the public claim preset
  - make JSONL-derived Kondo compute metrics single-source
  - expand tests from smoke coverage to claim coverage

### Thread E: External Research and Production Analogs

- Agent: `019db0d7-72f4-7ce3-b120-f431195ac229`
- Nickname: `Lorentz`
- Goal:
  - rank importable external ideas for a Kondo v2 beyond plain row cover
- Completion bar:
  - top-3 ideas with stack-specific importability judgment
- Current verdict:
  - reversible middle-layer drop/restore is the strongest future v2
  - Random-LTD is a control, not the path
  - MoD-style routed blocks are a later executor project

### Thread F: Skeptical Redteam

- Agent: `019db0d7-8054-7280-80d6-2a0108fce4af`
- Nickname: `Maxwell`
- Goal:
  - challenge the current Kondo story and identify the highest-leverage remaining optimization
- Completion bar:
  - certify the leading candidate is locally exhausted, or identify the real next optimization
- Current verdict:
  - not exhausted
  - remaining real headroom is executor-side, not more support approximations

## Shared Deliverables

Each thread must finish with:

1. A concrete recommendation or kill decision.
2. Exact file/result references.
3. A statement of what remains genuinely worth trying.
4. A statement of what should be frozen.
