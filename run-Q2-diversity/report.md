# Q2 Report: Diversity of Good Solutions at Equal Budget

- Budget (B) = 8192 objective evaluations per method per instance
- ε = 0.05 (good = value ≥ (1−ε)×optimum)
- Hamming cluster threshold d = ⌈n/4⌉
- 9 instances: n∈{12,16,20} × seeds∈{42,12345,67890}

## Table 1: Number of ε-good Hamming clusters (mean ± σ across 3 seeds)

| n | neal-multistart | QAOA p=1 | random+greedy |
|---|---|---|---|
| 12 | 1.33 ± 0.58 | 1.33 ± 0.58 | 1.67 ± 0.58 |
| 16 | 1.33 ± 0.58 | 1.67 ± 0.58 | 1.67 ± 0.58 |
| 20 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

## Table 2: Mean pairwise normalized Hamming distance among ε-good solutions

| n | neal-multistart | QAOA p=1 | random+greedy |
|---|---|---|---|
| 12 | 0.0831 ± 0.0829 | 0.0536 ± 0.0749 | 0.0603 ± 0.0824 |
| 16 | 0.0863 ± 0.0945 | 0.1050 ± 0.1297 | 0.0818 ± 0.0960 |
| 20 | 0.0359 ± 0.0448 | 0.0657 ± 0.0582 | 0.0571 ± 0.0515 |

## Verdict

**Cluster count summary (mean across all instances):**
- neal-multistart:  1.22
- QAOA p=1:         1.33
- random+greedy:    1.44

**Mean pairwise Hamming summary (mean across all instances):**
- neal-multistart:  0.0684
- QAOA p=1:         0.0747
- random+greedy:    0.0664

**Observation:** At equal budget of 8192 evaluations, 
all three methods produce a similar number of structurally different good solutions. 
No sampler shows a clear advantage in solution diversity at this instance scale. 
"Разницы нет" — a valid outcome for n ≤ 20. 


## Applicability to the Estimator (рекомендация №2)

The estimator's recommendation №2 calls for presenting 2–3 different sets of suppliers to the user rather than a single optimal one. The present experiment asks whether quantum or quantum-inspired samplers (QAOA / neal) are better suited for generating such diverse alternatives than classical random search with greedy repair, at a fixed evaluation budget. The results above suggest that — at the problem sizes tested (n ≤ 20) — the diversity of good solutions is largely similar across all three methods. Neither the QAOA distribution nor neal's simulated annealing produce substantially more structurally distinct high-quality solutions than simple random+greedy when given the same number of objective evaluations. This implies that, for the smaller subproblems encountered in the estimator's decomposition, a well-tuned classical multistart procedure can supply sufficient solution diversity without the overhead of quantum simulation. However, for larger n where QAOA's or neal's exploration may differ more markedly from random search, a re-evaluation would be warranted. If quantum hardware (QPU) becomes available, QAOA's native sampling from the full distribution could become a practical source of diverse candidates at no extra evaluation cost.
