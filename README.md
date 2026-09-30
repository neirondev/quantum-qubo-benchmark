# Quantum vs classical on QUBO — an honest benchmark

A small, reproducible benchmark that asks one uncomfortable question: **when does a quantum algorithm actually beat
classical computing?** Every number here is a real measurement; negative results are reported as results.

QUBO (minimize xᵀQx over binary x) is used as a common intermediate format: one objective and one evaluation formula
for every solver — greedy, simulated annealing (own implementation and D-Wave `neal`), gate-model QAOA (Qiskit Aer)
and brute force as the exact optimum. A separate line tests analog quantum hardware (QuEra Aquila on AWS Braket).

## Results

| Run | Question | Verdict |
|---|---|---|
| `run-005-qubo` | Where does thermal annealing beat greedy heuristics on subset-selection QUBOs? | **Confirmed:** annealing wins from n ≈ 500, +4–5 % value at n = 2000 |
| `run-Q1-micro-qaoa` | Does gate-model QAOA beat classical annealing on small QUBOs (n = 12, 16, 20)? | **No:** parity at n = 12, 3.5–10 pp worse at n = 16/20; runtime grows steeply with depth. A planned IonQ run was cancelled before spending the budget |
| `run-Q2-diversity` | Do samplers give more diverse good solutions than multistart? | **No difference:** ε-good solutions collapse into 1.0–1.7 clusters on these instances |
| `run-Q3-hopfield-aquila` | Can a Rydberg-blockade attractor ("pattern memory") be reproduced on real hardware? | **Yes:** stable 7–8-atom patterns reproduced on Aquila; the simulator–hardware gap is measured and explained |

**Q1 — approximation ratio vs exact optimum** (9 instances, 3 seeds per size):

| n | greedy | neal + repair | QAOA p=1 | QAOA p=2 |
|---|---|---|---|---|
| 12 | 0.932 | 1.000 | 1.000 | 0.991 |
| 16 | 0.965 | 0.988 | 0.913 | 0.953 |
| 20 | 0.916 | 0.982 | 0.888 | 0.881 |

QAOA runtime per instance: p=1 — 3–10 s, p=2 — 31–147 s (Qiskit Aer simulator).

**Q3 / H23 — QuEra Aquila:** exact target pattern in 0.581 (ring-8) and 0.573 (line-7) of hardware shots versus 0.95 and
0.90 on the ideal simulator; per-atom fidelity ≈ 0.934 / 0.924; total hardware cost $2.60. Pattern retention decays with
hold time, τ ≈ 42 µs (6.2σ) — a second loss channel beyond per-atom preparation/detection noise.

## Layout

- `run-005-qubo/` — QUBO generator (`qubo.mjs`), classical benchmark (greedy, own SA, `neal`), instances n ≤ 500.
- `run-Q1-micro-qaoa/` — QAOA p=1/p=2 (Qiskit Aer) vs `neal` + feasibility repair vs brute force; `report.md`.
- `run-Q2-diversity/` — diversity of good solutions: samplers vs multistart.
- `run-Q3-hopfield-aquila/` — Rydberg-blockade attractors: simulation, Aquila submissions and analysis; H23 retention curve.

Each run has its own `problem.yaml` and a result file (`RESULT.md` / `report.md`, in Russian; the table above is the English
summary). Task result files keep the Braket quantum-task ids; the AWS account id is replaced with `<account>`.

## Reproduce

```bash
pip install -r requirements.txt                    # Python 3.11+
cd run-005-qubo
for s in 42 12345 67890; do node qubo.mjs --export-qubo 2000 $s; done   # Node 18+; regenerates the n = 2000 instances
cd .. && shasum -a 256 -c run-005-qubo/LARGE_INSTANCES.sha256            # byte-for-byte check
```

The n = 2000 instances (≈ 90 MB each) are not stored in the repository: the generator is deterministic and reproduces
them byte for byte from the seed (checksums in `run-005-qubo/LARGE_INSTANCES.sha256`). Aquila submissions require an AWS
account with Braket access and cost real money; everything else runs locally for free.

## Status

Research code, published as measured. Planned next steps (not done yet): retention curves and new geometries on Aquila,
and a cleaned-up release of the QUBO generator, solver bridges and evaluation formula.

## License

MIT — see `LICENSE`.
