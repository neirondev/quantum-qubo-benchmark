#!/usr/bin/env python3
"""Q3 фаза 2: сабмит ДВУХ задач на QuEra Aquila (согласие Дениса получено 2026-09-04).
Дизайн — по gate-notes-phase1.md:
  Task A: кольцо-8 (сосед ~5.85 мкм, V_nn≈1.35e8 > Δ_end) — номинальный протокол.
  Task B: нечётная цепочка P7 (шаг 6.0 мкм), Δ_end=1.1e8 — единственное целевое MIS.
100 шотов на задачу. Смета: 2 × ($0.30 + 100×$0.01) = $2.60. Spending limit $5.
Скрипт ТОЛЬКО отправляет и записывает ARN'ы (окно исполнения — пятница 04:00 UTC),
результаты забирает phase2_fetch.py позже.
"""
import json
import math
import time

from braket.aws import AwsDevice
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement

import experiment as ex  # проверенные строительные блоки фазы 1

DEVICE_ARN = 'arn:aws:braket:us-east-1::device/qpu/quera/Aquila'
SHOTS = 100
PER_TASK, PER_SHOT = 0.30, 0.01
COST_GUARD = 3.00  # $ на весь сабмит

CENTER = (37.5e-6, 38.0e-6)  # центр поля 75x76 мкм
RES = 1e-7                    # разрешение позиций 0.1 мкм


def snap(v):
    return round(v / RES) * RES


def ring8(neighbor=5.85e-6):
    r = neighbor / (2 * math.sin(math.pi / 8))
    reg = AtomArrangement()
    for k in range(8):
        # поворот на pi/8: ряды по y = {±2.9, ±7.1} мкм, зазоры 4.1/5.8 >= 4 мкм
        # (квирк Aquila: y-разнесение сайтов 0 или >=4 мкм; симулятор это не проверял)
        a = math.pi / 8 + 2 * math.pi * k / 8
        reg.add((snap(CENTER[0] + r * math.cos(a)), snap(CENTER[1] + r * math.sin(a))))
    return reg


def line7(spacing=6.0e-6):
    x0 = CENTER[0] - 3 * spacing
    reg = AtomArrangement()
    for i in range(7):
        reg.add((snap(x0 + i * spacing), snap(CENTER[1])))
    return reg


def main():
    est = 2 * (PER_TASK + SHOTS * PER_SHOT)
    assert est <= COST_GUARD, f'смета ${est:.2f} превышает гард ${COST_GUARD}'
    print(f'Смета: ${est:.2f} (гард ${COST_GUARD}, spending limit $5) — ок')

    dev = AwsDevice(DEVICE_ARN)
    print('Статус устройства:', dev.status)

    tasks = []
    plans = [
        ('ring8-nominal', ring8(), ex.build_driving_field()),               # Δ_end=1.2e8 < V_nn(5.85µm)
        ('line7-det1.1e8', line7(), ex.build_driving_field(det_end=1.1e8)),  # ниже V_nn(6.0µm)=1.16e8
    ]
    for name, reg, drive in plans:
        prog = AnalogHamiltonianSimulation(register=reg, hamiltonian=drive)
        prog = prog.discretize(dev)  # снап к разрешениям железа
        task = dev.run(prog, shots=SHOTS)
        meta = {
            'name': name,
            'arn': task.id,
            'shots': SHOTS,
            'submitted_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'state_on_submit': task.state(),
        }
        print(f'{name}: {task.id} → {meta["state_on_submit"]}')
        tasks.append(meta)

    with open('phase2-tasks.json', 'w') as f:
        json.dump(tasks, f, indent=1, ensure_ascii=False)
    print('ARN-ы сохранены в phase2-tasks.json; результаты — после окна (пятница 04:00 UTC).')


if __name__ == '__main__':
    main()
