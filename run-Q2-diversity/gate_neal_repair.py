#!/usr/bin/env python3
"""Гейт-фикс Q1: neal + ОБЯЗАТЕЛЬНЫЙ репайр (в прогоне агента neal шёл без репайра
и все 9 решений вышли за бюджет). Оценка — тем же портом, что валидирован
бит-в-бит против qubo.mjs (импорт из qaoa_experiment).
Репайр как в UI-005: пока бюджет нарушен — выкинуть выбранный элемент с худшим
value/cost; затем жадно докинуть влезающие по value/cost.
"""
import json
import glob

import neal
from qaoa_experiment import load_qubo_json, evaluate_objective, check_feasible


def repair(graph, bits):
    bits = list(bits)
    vals, costs, budget = graph['values'], graph['costs'], graph['budget']
    feas, used = check_feasible(graph, bits)
    while not feas:
        chosen = [i for i, b in enumerate(bits) if b]
        worst = min(chosen, key=lambda i: vals[i] / max(costs[i], 1e-9))
        bits[worst] = 0
        feas, used = check_feasible(graph, bits)
    order = sorted(range(len(bits)), key=lambda i: -vals[i] / max(costs[i], 1e-9))
    for i in order:
        if not bits[i] and used + costs[i] <= budget + 1e-9:
            bits[i] = 1
            used += costs[i]
    return bits


def neal_solve(qubo_path, num_reads=64):
    Q_entries, graph = load_qubo_json(qubo_path)
    Q = {}
    for i, j, w in Q_entries:
        Q[(i, j)] = Q.get((i, j), 0.0) + w
    sampler = neal.SimulatedAnnealingSampler()
    ss = sampler.sample_qubo(Q, num_reads=num_reads)
    best_obj, best_bits = -1e300, None
    n = len(graph['values'])
    for sample, _ in ss.data(['sample', 'energy']):
        bits = [int(sample.get(i, 0)) for i in range(n)]  # ключи int!
        bits = repair(graph, bits)
        obj = evaluate_objective(graph, bits)
        if obj > best_obj:
            best_obj, best_bits = obj, bits
    feas, _ = check_feasible(graph, best_bits)
    assert feas, 'после репайра решение обязано быть feasible'
    return best_obj


def main():
    res = json.load(open('results-q1.json'))
    opt = {(r['n'], r['seed']): r['best_value'] for r in res['results'] if r['solver'] == 'brute'}
    rows = []
    for path in sorted(glob.glob('qubo-*.json')):
        _, n, seed = path.replace('.json', '').split('-')
        n, seed = int(n), int(seed)
        obj = neal_solve(path)
        ar = obj / opt[(n, seed)]
        rows.append({'n': n, 'seed': seed, 'solver': 'neal+repair',
                     'best_value': obj, 'optimum': opt[(n, seed)],
                     'approx_ratio': round(ar, 6), 'feasible': True})
        print(f'n={n} seed={seed}: neal+repair={obj} vs brute={opt[(n, seed)]} → ar={ar:.3f}')
    with open('results-neal-repaired.json', 'w') as f:
        json.dump(rows, f, indent=1)
    print('записано results-neal-repaired.json')


if __name__ == '__main__':
    main()
