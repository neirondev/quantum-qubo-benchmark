#!/usr/bin/env python3
"""Гейт-проверка: какие паттерны РЕАЛЬНО доминируют на номинале у линии и кластеров.
Гипотеза контролёра: выигрывают максимальные независимые множества (MIS) открытой
цепочки с минимальными хвостами вторых соседей, а не строгая альтернация.
Запуск: arch -arm64 /tmp/aquila_venv/bin/python gate_check_patterns.py
"""
import experiment as ex
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from collections import Counter

sim = ex.LocalSimulator('braket_ahs')

def top_patterns(geo_name, shots=100):
    register = ex.GEOMETRIES[geo_name]()
    drive = ex.build_driving_field()  # номинал
    task = sim.run(AnalogHamiltonianSimulation(register=register, hamiltonian=drive), shots=shots)
    seqs = [tuple(m.post_sequence) for m in task.result().measurements]
    return Counter(seqs)

for geo in ('line', 'clusters'):
    c = top_patterns(geo)
    print(f'=== {geo} (номинал, 100 шотов; 0=Rydberg, 1=ground) ===')
    for pat, n in c.most_common(6):
        ryd = [i + 1 for i, b in enumerate(pat) if b == 0]
        print(f'  {"".join(map(str, pat))}  ×{n:3d}   Rydberg на позициях {ryd}')
    print()
