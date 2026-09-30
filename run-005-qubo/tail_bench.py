import json, subprocess, time
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def sh(cmd, timeout=1800): return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
def evaluate(qf, bf):
    r = sh(f"node qubo.mjs --evaluate {qf} {bf}"); t = r.stdout.strip()
    return json.loads(t[t.find('{'):])
def repair(qf, bits):
    d = json.load(open(qf)); g = d["graph"]; vals, costs, budget = g["values"], g["costs"], g["budget"]
    x = list(bits)
    used = lambda: sum(c for c, xi in zip(costs, x) if xi)
    while used() > budget:
        wi = min((i for i, xi in enumerate(x) if xi), key=lambda i: vals[i]/costs[i]); x[wi] = 0
    for i in sorted(range(len(x)), key=lambda i: -vals[i]/costs[i]):
        if not x[i] and used() + costs[i] <= budget: x[i] = 1
    return x
res = []
for n, seed, solvers in [(500, 67890, ["neal"]), (2000, 42, ["greedy", "neal"]), (2000, 12345, ["greedy", "neal"])]:
    qf = f"qubo-{n}-{seed}.json"; sh(f"node qubo.mjs --export-qubo {n} {seed}")
    for solver in solvers:
        bf = f"{solver}-{n}-{seed}.json"; t0 = time.time()
        if solver == "neal": sh(f"python3 neal_solve.py {qf} {bf}")
        else: sh(f"node qubo.mjs --solve-{solver} {qf} {bf}")
        bits = json.load(open(bf))["bits"]; json.dump({"bits": repair(qf, bits)}, open(f"r-{bf}", "w"))
        ev = evaluate(qf, f"r-{bf}")
        res.append({"n": n, "seed": seed, "solver": solver, "obj": ev["objective"], "feas": ev["feasible"], "s": round(time.time()-t0)})
        print(res[-1], flush=True)
json.dump(res, open("results-phase2-tail.json", "w"), indent=1)
print("TAIL-DONE")
