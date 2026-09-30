#!/usr/bin/env python3
"""neal_solve.py — решает QUBO из JSON-файла (формат qubo-N-SEED.json) через
   neal.SimulatedAnnealingSampler, сохраняет битовую строку в JSON.

Usage:
    arch -arm64 python3 neal_solve.py qubo-N-SEED.json output-bits.json
"""
import sys
import json
import time

try:
    import neal
except ImportError:
    print(json.dumps({"error": "neal not installed. pip install dwave-neal"}))
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: neal_solve.py qubo.json output.json", file=sys.stderr)
        sys.exit(1)

    qubo_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(qubo_path, "r") as f:
        data = json.load(f)

    n = data["n"]
    entries = data["Q"]  # list of [i, j, w]

    # Build Q matrix as dict for dimod
    Q = {}
    for i, j, w in entries:
        Q[(int(i), int(j))] = float(w)

    # Run neal
    t0 = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=64, num_sweeps=2000)
    best = sampleset.first
    runtime_ms = (time.time() - t0) * 1000

    # Extract bits
    bits = [int(best.sample.get(i, 0)) for i in range(n)]

    output = {
        "bits": bits,
        "energy": float(best.energy),
        "runtime_ms": round(runtime_ms, 3)
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output))


if __name__ == "__main__":
    main()