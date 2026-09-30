#!/usr/bin/env python3
"""qaoa_experiment.py — Q1: micro-QAOA vs classical solvers on small QUBO instances.

Usage:
    arch -arm64 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 qaoa_experiment.py

Generates results-q1.json incrementally.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ── paths & globals ──────────────────────────────────────────────────────
BASE = Path(__file__).parent.resolve()
PYTHON = "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
NODE = "node"
QUBO_MJS = BASE / "qubo.mjs"
NEAL_SOLVE = BASE / "neal_solve.py"
RESULTS_FILE = BASE / "results-q1.json"

NS = [12, 16, 20]
SEEDS = [42, 12345, 67890]

# ── helpers: subprocess with arch -arm64 ─────────────────────────────────
def run_python(args, cwd=None, timeout=300):
    """Run a Python script with arch -arm64."""
    cmd = ["arch", "-arm64", PYTHON] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=cwd or BASE, timeout=timeout)

def run_node(args, cwd=None, timeout=30):
    cmd = [NODE] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=cwd or BASE, timeout=timeout)


# ── QUBO generation ──────────────────────────────────────────────────────
def generate_qubo(n, seed, out_dir=None):
    """Generate QUBO JSON file using qubo.mjs --export-qubo.
    Returns the Path to the generated file.
    """
    out_dir = out_dir or BASE
    fname = out_dir / f"qubo-{n}-{seed}.json"
    if fname.exists():
        return fname
    result = run_node([str(QUBO_MJS), "--export-qubo", str(n), str(seed)])
    if result.returncode != 0:
        raise RuntimeError(f"qubo.mjs --export-qubo failed: {result.stderr}")
    return fname


# ── evaluation port (verify bit-against qubo.mjs --evaluate) ────────────
def evaluate_objective(graph, bits):
    """Port of qubo.mjs evaluateObjectiveFromX.
    graph: dict with keys 'values', 'costs', 'budget', 'edges'.
    bits: list of 0/1 ints.
    Returns the problem objective value (higher = better).
    """
    val = 0
    for i, v in enumerate(graph["values"]):
        if bits[i]:
            val += v
    for e in graph["edges"]:
        if bits[e["i"]] and bits[e["j"]]:
            if e["type"] == "contradiction":
                val -= e["weight"]
            else:
                val += e["weight"]
    return val


def check_feasible(graph, bits):
    """Check if budget is satisfied."""
    used = sum(graph["costs"][i] for i, b in enumerate(bits) if b)
    return used <= graph["budget"] + 1e-9, used


def load_qubo_json(path):
    """Load qubo JSON and return (Q_entries, graph_dict)."""
    with open(path) as f:
        data = json.load(f)
    graph = {
        "n": data["n"],
        "values": data["graph"]["values"],
        "costs": data["graph"]["costs"],
        "budget": data["graph"]["budget"],
        "edges": data["graph"]["edges"],
    }
    Q_entries = data["Q"]  # list of [i, j, w]
    return Q_entries, graph


def verify_evaluation():
    """Test the Python port of evaluate_objective against qubo.mjs --evaluate."""
    print("[VERIFY] Checking evaluation port on 3 solutions...")
    test_cases = [
        (BASE / "qubo-12-42.json", [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1]),
        (BASE / "qubo-12-42.json", [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]),
        (BASE / "qubo-12-42.json", [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    ]
    for qpath, bits in test_cases:
        # Write bits to temp file
        bits_file = BASE / "_verify_bits.json"
        with open(bits_file, "w") as f:
            json.dump({"bits": bits}, f)
        # Evaluate via qubo.mjs
        result = run_node([str(QUBO_MJS), "--evaluate", str(qpath), str(bits_file)])
        mjs = json.loads(result.stdout)
        # Evaluate via Python port
        _, graph = load_qubo_json(qpath)
        py_val = evaluate_objective(graph, bits)
        # Compare
        assert abs(py_val - mjs["objective"]) < 1e-9, \
            f"Mismatch: mjs={mjs['objective']}, py={py_val}"
        print(f"  OK: objective={py_val}")
    os.unlink(bits_file)
    print("[VERIFY] Evaluation port: PASSED")


# ── QUBO → Ising conversion ────────────────────────────────────────────
def qubo_to_ising(Q_entries, n):
    """Convert QUBO matrix entries (list of [i,j,w]) to Ising coefficients.

    Returns:
        constant: float — energy offset
        h: list[n] of float — linear Z coefficients
        J: dict[(i,j)] of float — quadratic ZZ coefficients (i < j)
    """
    # Build full Q matrix
    Q = np.zeros((n, n), dtype=float)
    for i, j, w in Q_entries:
        Q[i][j] = w
    # Symmetrize (Q should already be symmetric, but just in case)
    # The QUBO uses Q[i][j] for i≤j
    constant = 0.0
    h = np.zeros(n, dtype=float)
    J = {}

    for i in range(n):
        constant += Q[i][i] / 2.0
        h[i] -= Q[i][i] / 2.0
        for j in range(i + 1, n):
            constant += Q[i][j] / 4.0
            h[i] -= Q[i][j] / 4.0
            h[j] -= Q[i][j] / 4.0
            J[(i, j)] = Q[i][j] / 4.0

    return constant, h.tolist(), J


def ising_energy(bits, constant, h, J):
    """Compute Ising energy E = constant + sum h_i*z_i + sum J_ij*z_i*z_j.
    bits: list of 0/1, z_i = 1 - 2*bits[i]
    """
    z = [1 - 2 * b for b in bits]
    E = constant
    for i, hi in enumerate(h):
        E += hi * z[i]
    for (i, j), Jij in J.items():
        E += Jij * z[i] * z[j]
    return E


# ── QAOA circuit ────────────────────────────────────────────────────────
def build_qaoa_circuit(n, gamma_layer, beta_layer, h, J):
    """Build a QAOA circuit with given angles.
    gamma_layer: list of gamma values for each layer (length p)
    beta_layer: list of beta values for each layer (length p)

    Returns QuantumCircuit with all qubits measured.
    """
    p = len(gamma_layer)
    qc = QuantumCircuit(n)
    # Initial state
    qc.h(range(n))

    for layer in range(p):
        gamma = gamma_layer[layer]
        beta = beta_layer[layer]
        # Phase separator: e^{-i*gamma*H_P}
        # Linear terms: RZ(2*gamma*h_i)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * gamma * h[i], i)
        # Quadratic terms: CX + RZ + CX
        for (i, j), Jij in J.items():
            if abs(Jij) > 1e-12:
                qc.cx(i, j)
                qc.rz(2 * gamma * Jij, j)
                qc.cx(i, j)
        # Mixer: e^{-i*beta*X}
        qc.rx(2 * beta, range(n))

    qc.measure_all()
    return qc


# ── QAOA grid search ────────────────────────────────────────────────────
def qaoa_counts(qc, sim, shots):
    """Run transpiled QAOA circuit on Aer and return counts dict."""
    try:
        transpiled = transpile(qc, sim)
        result = sim.run(transpiled, shots=shots).result()
        return result.get_counts()
    except Exception as e:
        print(f"    [ERROR] QAOA run failed: {e}")
        return {}


def mean_ising_energy(counts, constant, h, J):
    """Compute mean Ising energy across shots."""
    total = 0
    n_shots = sum(counts.values())
    if n_shots == 0:
        return 0.0
    for bits_str, count in counts.items():
        # qiskit counts: key is little-endian (bit[-1] = qubit 0)
        b = [int(c) for c in bits_str[::-1]]  # now b[i] = qubit i
        total += ising_energy(b, constant, h, J) * count
    return total / n_shots


def best_sample(counts, constant, h, J, graph):
    """From counts, find the sample with lowest Ising energy
    and return (bits_list, ising_energy, problem_objective).
    """
    best_bits = None
    best_ising = float("inf")
    for bits_str, _ in counts.items():
        b = [int(c) for c in bits_str[::-1]]
        E = ising_energy(b, constant, h, J)
        if E < best_ising:
            best_ising = E
            best_bits = b
    obj = evaluate_objective(graph, best_bits)
    return best_bits, best_ising, obj


def best_feasible_objective(counts, graph):
    """From QAOA counts, find the highest problem objective
    among samples that satisfy the budget constraint.
    Returns (best_obj, feasible, p_optimal).
    Also checks if the optimal bitstring appears.
    """
    best_obj = -1e300
    found_feasible = False
    for bits_str, _ in counts.items():
        b = [int(c) for c in bits_str[::-1]]
        obj = evaluate_objective(graph, b)
        feas, _ = check_feasible(graph, b)
        if feas and obj > best_obj:
            best_obj = obj
            found_feasible = True
    if not found_feasible:
        # Fall back to best overall (may be infeasible)
        for bits_str, _ in counts.items():
            b = [int(c) for c in bits_str[::-1]]
            obj = evaluate_objective(graph, b)
            if obj > best_obj:
                best_obj = obj
        return best_obj, False, 0
    return best_obj, True, 0


GRID_8 = np.linspace(0.1, 1.5, 8)


def grid_search_p1(n, constant, h, J, graph, sim):
    """p=1 QAOA: 8x8 grid over gamma, beta.
    Returns (best_gamma, best_beta, best_mean_E, final_counts).
    """
    print("    [QAOA p=1] Grid search (8x8)...")
    best_gamma, best_beta, best_mean = None, None, float("inf")
    for g in GRID_8:
        for b in GRID_8:
            qc = build_qaoa_circuit(n, [g], [b], h, J)
            counts = qaoa_counts(qc, sim, shots=1024)
            if not counts:
                continue
            mean_E = mean_ising_energy(counts, constant, h, J)
            if mean_E < best_mean:
                best_mean = mean_E
                best_gamma, best_beta = g, b

    print(f"    [QAOA p=1] Best angles: gamma={best_gamma:.3f}, beta={best_beta:.3f}, mean_E={best_mean:.3f}")
    # Final run
    qc = build_qaoa_circuit(n, [best_gamma], [best_beta], h, J)
    counts = qaoa_counts(qc, sim, shots=8192)
    return best_gamma, best_beta, best_mean, counts


def grid_search_p2(n, constant, h, J, graph, sim, p1_gamma, p1_beta):
    """p=2 QAOA: 5x5 local grid around p1_best angles for each layer.
    Grid: gamma ∈ [p1_gamma - 0.5, p1_gamma + 0.5] × 5
          beta ∈ [p1_beta - 0.5, p1_beta + 0.5] × 5
    Total: 5^4 = 625 points.
    Returns (best_angles, best_mean_E, final_counts).
    """
    print("    [QAOA p=2] Local grid search (5x5 per layer = 625 points)...")

    g_grid = np.linspace(max(0.01, p1_gamma - 0.5), p1_gamma + 0.5, 5)
    b_grid = np.linspace(max(0.01, p1_beta - 0.5), p1_beta + 0.5, 5)

    best_gamma1, best_beta1 = None, None
    best_gamma2, best_beta2 = None, None
    best_mean = float("inf")

    for g1 in g_grid:
        for b1 in b_grid:
            for g2 in g_grid:
                for b2 in b_grid:
                    qc = build_qaoa_circuit(n, [g1, g2], [b1, b2], h, J)
                    counts = qaoa_counts(qc, sim, shots=1024)
                    if not counts:
                        continue
                    mean_E = mean_ising_energy(counts, constant, h, J)
                    if mean_E < best_mean:
                        best_mean = mean_E
                        best_gamma1, best_beta1 = g1, b1
                        best_gamma2, best_beta2 = g2, b2

    print(f"    [QAOA p=2] Best: γ1={best_gamma1:.3f}, β1={best_beta1:.3f}, γ2={best_gamma2:.3f}, β2={best_beta2:.3f}, mean_E={best_mean:.3f}")
    # Final run
    qc = build_qaoa_circuit(n, [best_gamma1, best_gamma2], [best_beta1, best_beta2], h, J)
    counts = qaoa_counts(qc, sim, shots=8192)
    return (best_gamma1, best_beta1, best_gamma2, best_beta2), best_mean, counts


# ── Brute force ─────────────────────────────────────────────────────────
def brute_force_optimum(graph):
    """Exact enumeration of all 2^n assignments.
    Pure Python loop, fast enough for n ≤ 20.
    Returns (best_bits, best_objective).
    """
    n = graph["n"]
    values = graph["values"]
    costs = graph["costs"]
    budget = graph["budget"]
    edges = graph["edges"]

    total = 1 << n
    best_obj = -1e300
    best_idx = -1

    for idx in range(total):
        # Decode bits from integer idx
        # Budget check (fast rejection)
        used = 0
        obj_val = 0
        ok = True
        # We'll decode bits and compute objective simultaneously
        # First pass: compute cost and value contribution
        x_bits = [0] * n
        for k in range(n):
            bit = (idx >> k) & 1
            x_bits[k] = bit
            if bit:
                used += costs[k]
                obj_val += values[k]
        if used > budget + 1e-9:
            continue

        # Edge contributions
        for e in edges:
            if x_bits[e["i"]] and x_bits[e["j"]]:
                if e["type"] == "synergy":
                    obj_val += e["weight"]
                else:
                    obj_val -= e["weight"]

        if obj_val > best_obj:
            best_obj = obj_val
            best_idx = idx

    best_bits = [(best_idx >> k) & 1 for k in range(n)]
    return best_bits, float(best_obj)


# ── Greedy solver via qubo.mjs ──────────────────────────────────────────
def solve_greedy(qubo_path):
    """Run qubo.mjs --solve-greedy and return bits list."""
    bits_file = BASE / "_greedy_bits.json"
    result = run_node([str(QUBO_MJS), "--solve-greedy", str(qubo_path), str(bits_file)])
    if result.returncode != 0:
        raise RuntimeError(f"greedy failed: {result.stderr}")
    with open(bits_file) as f:
        data = json.load(f)
    os.unlink(bits_file)
    return data["bits"]


# ── Neal solver ─────────────────────────────────────────────────────────
def solve_neal(qubo_path):
    """Run neal_solve.py and return (bits, runtime_ms)."""
    bits_file = BASE / "_neal_bits.json"
    result = run_python([str(NEAL_SOLVE), str(qubo_path), str(bits_file)])
    if result.returncode != 0:
        raise RuntimeError(f"neal_solve failed: {result.stderr}")
    with open(bits_file) as f:
        data = json.load(f)
    os.unlink(bits_file)
    return data["bits"], data["runtime_ms"]


# ── Results ─────────────────────────────────────────────────────────────
def load_results():
    """Load existing results file, or return empty dict."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"results": [], "meta": {"ns": NS, "seeds": SEEDS, "generated_at": None}}


def save_results(results):
    """Write results to file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVE] Written {RESULTS_FILE}")


def make_result(n, seed, solver, best_value, optimum, wall_time_s, p_optimal_in_shots=None, feasible=True):
    """Create a result dict with proper approx_ratio handling.
    Infeasible solutions get approx_ratio = 0.
    """
    ratio = best_value / optimum if (optimum != 0 and feasible) else 0.0
    return {
        "n": n, "seed": seed, "solver": solver,
        "best_value": best_value, "optimum": optimum,
        "approx_ratio": round(ratio, 6),
        "feasible": feasible,
        "p_optimal_in_shots": p_optimal_in_shots,
        "wall_time_s": round(wall_time_s, 4)
    }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Q1 Experiment: micro-QAOA vs classical solvers")
    print("=" * 60)

    # Verify evaluation port first
    verify_evaluation()

    sim = AerSimulator()
    results = load_results()
    seen = set()
    for r in results["results"]:
        seen.add((r["n"], r["seed"], r["solver"]))

    for n in NS:
        for seed in SEEDS:
            print(f"\n{'─' * 50}")
            print(f"[n={n}, seed={seed}]")
            print(f"{'─' * 50}")

            t0_total = time.time()

            # 1. Generate / load QUBO
            qubo_path = generate_qubo(n, seed)
            Q_entries, graph = load_qubo_json(qubo_path)

            # Convert to Ising
            constant, h, J = qubo_to_ising(Q_entries, n)

            # 2. Brute force (optimum) — always compute locally for opt_bits/opt_val
            # Only save to results if not already done
            opt_bits, opt_val = None, None
            for r in results["results"]:
                if r["n"] == n and r["seed"] == seed and r["solver"] == "brute":
                    opt_val = r["optimum"]
                    break
            if opt_val is not None:
                print(f"  [Brute force] Already done: optimum={opt_val}, recomputing bits for p_optimal check...")
            print("  [Brute force] Enumerating...")
            t0 = time.time()
            opt_bits, opt_val_computed = brute_force_optimum(graph)
            wall_time = time.time() - t0
            print(f"  [Brute force] Optimum = {opt_val_computed}, wall={wall_time:.2f}s")
            if opt_val is None:
                opt_val = opt_val_computed
                results["results"].append(make_result(
                    n, seed, "brute", opt_val, opt_val, wall_time, feasible=True))
                save_results(results)
                seen.add((n, seed, "brute"))

            # 3. Greedy
            key = (n, seed, "greedy")
            if key not in seen:
                print("  [Greedy] Solving...")
                t0 = time.time()
                bits_g = solve_greedy(qubo_path)
                wall_time = time.time() - t0
                obj_g = evaluate_objective(graph, bits_g)
                feasible, _ = check_feasible(graph, bits_g)
                if not feasible:
                    print(f"  [Greedy] WARNING: infeasible solution, objective={obj_g}")
                ar = obj_g / opt_val if opt_val != 0 else 0.0
                print(f"  [Greedy] Objective={obj_g}, approx_ratio={ar:.4f}, wall={wall_time:.2f}s")
                results["results"].append(make_result(
                    n, seed, "greedy", obj_g, opt_val, wall_time, feasible=feasible))
                save_results(results)
                seen.add(key)
            else:
                print("  [Greedy] Already done, skipping.")

            # 4. Neal
            key = (n, seed, "neal")
            if key not in seen:
                print("  [Neal] Solving...")
                t0 = time.time()
                bits_n, rt_ms = solve_neal(qubo_path)
                wall_time = time.time() - t0
                obj_n = evaluate_objective(graph, bits_n)
                feasible, _ = check_feasible(graph, bits_n)
                if not feasible:
                    print(f"  [Neal] WARNING: infeasible solution, objective={obj_n}")
                ar = obj_n / opt_val if opt_val != 0 else 0.0
                print(f"  [Neal] Objective={obj_n}, approx_ratio={ar:.4f}, wall={wall_time:.2f}s")
                results["results"].append(make_result(
                    n, seed, "neal", obj_n, opt_val, wall_time, feasible=feasible))
                save_results(results)
                seen.add(key)
            else:
                print("  [Neal] Already done, skipping.")

            # 5. QAOA p=1
            key = (n, seed, "qaoa-p1")
            if key not in seen:
                print("  [QAOA p=1] Running...")
                t0 = time.time()
                p1_gamma, p1_beta, p1_mean_E, p1_counts = grid_search_p1(n, constant, h, J, graph, sim)
                # Check if optimal bitstring appears in shots
                opt_str = "".join(str(b) for b in opt_bits[::-1])  # qiskit endianness
                p_optimal = 1 if opt_str in p1_counts else 0
                # Best feasible objective
                best_obj_in_shots, feasible, _ = best_feasible_objective(p1_counts, graph)
                wall_time = time.time() - t0
                ar = best_obj_in_shots / opt_val if (opt_val != 0 and feasible) else 0.0
                print(f"  [QAOA p=1] Best obj in shots={best_obj_in_shots}, approx_ratio={ar:.4f}, P(optimal)={p_optimal}, feasible={feasible}")
                results["results"].append(make_result(
                    n, seed, "qaoa-p1", best_obj_in_shots, opt_val, wall_time,
                    p_optimal_in_shots=p_optimal, feasible=feasible))
                save_results(results)
                seen.add(key)
            else:
                print("  [QAOA p=1] Already done, skipping.")

            # 6. QAOA p=2 — always starts fresh (needs p1 angles)
            # Always compute p1 grid to get angles for p2 initialization
            key = (n, seed, "qaoa-p2")
            if key not in seen:
                print("  [QAOA p=2] Running p=1 grid search for angle initialization...")
                p1_gamma, p1_beta, p1_mean_E, _ = grid_search_p1(n, constant, h, J, graph, sim)

                print(f"  [QAOA p=2] Running with p1 angles: gamma={p1_gamma:.3f}, beta={p1_beta:.3f}")
                t0 = time.time()
                p2_angles, p2_mean_E, p2_counts = grid_search_p2(n, constant, h, J, graph, sim, p1_gamma, p1_beta)
                opt_str = "".join(str(b) for b in opt_bits[::-1])
                p_optimal = 1 if opt_str in p2_counts else 0
                best_obj_in_shots, feasible, _ = best_feasible_objective(p2_counts, graph)
                wall_time = time.time() - t0
                ar = best_obj_in_shots / opt_val if (opt_val != 0 and feasible) else 0.0
                print(f"  [QAOA p=2] Best obj in shots={best_obj_in_shots}, approx_ratio={ar:.4f}, P(optimal)={p_optimal}, feasible={feasible}")
                results["results"].append(make_result(
                    n, seed, "qaoa-p2", best_obj_in_shots, opt_val, wall_time,
                    p_optimal_in_shots=p_optimal, feasible=feasible))
                save_results(results)
                seen.add(key)
            else:
                print("  [QAOA p=2] Already done, skipping.")

    # Update meta
    results["meta"]["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_results(results)

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()