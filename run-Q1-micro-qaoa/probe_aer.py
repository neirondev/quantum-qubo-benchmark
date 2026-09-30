#!/usr/bin/env python3
"""Проверенная проба Aer для QAOA (фаза-0 сниппет, не переоткрывать API).
Интерпретатор с qiskit 2.5.2 + qiskit-aer:
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3

ГЛАВНЫЙ УРОК (получен живым прогоном): при произвольных углах p=1 QAOA
концентрируется где попало (на пути P3 углы 0.8/0.4 дали ХУДШИЕ состояния
000/111). Углы (gamma, beta) ОБЯЗАТЕЛЬНО оптимизировать — здесь сеткой,
в эксперименте — сеткой или COBYLA. Без оптимизации углов сравнение
QAOA с классикой нечестно.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

EDGES = [(0, 1), (1, 2)]  # путь P3; MaxCut: минимизируем E = sum z_i z_j
N = 3
sim = AerSimulator()


def qaoa_counts(gamma: float, beta: float, shots: int = 1024):
    qc = QuantumCircuit(N)
    qc.h(range(N))
    for (i, j) in EDGES:                 # e^{-i gamma Z_i Z_j}
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)
    qc.rx(2 * beta, range(N))            # mixer e^{-i beta X}
    qc.measure_all()
    return sim.run(transpile(qc, sim), shots=shots).result().get_counts()


def energy(bits_q2q1q0: str) -> int:
    # ключи counts в little-endian qiskit: символ [-1] = q0. Разворачиваем.
    b = bits_q2q1q0[::-1]                # теперь b[i] = кубит i
    z = [1 - 2 * int(x) for x in b]
    return sum(z[i] * z[j] for i, j in EDGES)


def mean_energy(counts) -> float:
    tot = sum(counts.values())
    return sum(energy(k) * v for k, v in counts.items()) / tot


# сетка углов — тот же приём обязателен в эксперименте
grid = np.linspace(0.1, 1.5, 8)
best_angles, best_E = None, 1e9
for g in grid:
    for b in grid:
        E = mean_energy(qaoa_counts(g, b, shots=512))
        if E < best_E:
            best_E, best_angles = E, (g, b)

counts = qaoa_counts(*best_angles, shots=4096)
top = max(counts, key=counts.get)
print(f"лучшие углы: gamma={best_angles[0]:.2f} beta={best_angles[1]:.2f}  <E>={best_E:.3f}")
print("top-3:", sorted(counts.items(), key=lambda kv: -kv[1])[:3])
assert top[::-1] in ("010", "101"), f"ожидали антикоррелированный оптимум, получили {top}"
assert best_E < -1.0, f"<E> при лучших углах должен приближаться к -2, получено {best_E}"
print("OK: Aer работает, QAOA-шаблон с оптимизацией углов корректен")
