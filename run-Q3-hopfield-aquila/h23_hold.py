#!/usr/bin/env python3
"""H23: проверка гипотезы №23 — распадается ли паттерн ВО ВРЕМЕНИ.

Переформулировка (кандидат предлагал менять интервал МЕЖДУ шотами — этим в Braket AHS
не управляют, атомы каждый шот перезагружаются). Управляемая версия: после свипа
УДЕРЖИВАТЬ финальное поле t_hold, затем считывать. Если retention падает с ростом
t_hold — распад во времени реален; если плоско в пределах шума — держится модель
пер-атомного шума подготовки/детекции (наш вывод Q3).

Точка t_hold=0 берётся из УЖЕ оплаченных данных (ring8-nominal, 0.581 на 93 загруженных
шотах) — протокол идентичен, добавляется только удержание.

Запуск: --local (валидация, бесплатно) | --submit (2 задачи на Aquila, $2.00)
"""
import json
import sys
import time

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries

import experiment as ex  # проверенные блоки фазы 1 (геометрии, константы, retention)

HOLDS = [6e-6, 15e-6]      # прогон 1 (выполнен): 2 × ($0.30 + 70×$0.01) = $2.00
SHOTS = 70
# Прогон 2 (добор мощности, лимит поднят до $30): точки подобраны, чтобы зажать
# экспоненту вместе с уже снятыми 0/6/15. 15.5 мкс — у предела длительности 20 мкс.
HOLDS2 = [2e-6, 8e-6, 12e-6, 15.5e-6]
SHOTS2 = 200                # 4 × ($0.30 + 200×$0.01) = $9.20
PER_TASK, PER_SHOT = 0.30, 0.01
LIMIT_REMAINING = 25.40     # $30 лимит − $4.60 потрачено


def build_drive_with_hold(t_hold, amp_max=ex.RABI_MAX, det_start=ex.DET_START,
                          det_end=ex.DET_END_NOMINAL, t_sweep=ex.T_SWEEP_NOMINAL,
                          t_ramp=ex.T_RAMP):
    """Номинальный свип + удержание финального поля t_hold (амплитуда 0, detuning держится)."""
    t_end = t_sweep + t_hold

    amp_ts = TimeSeries()
    amp_ts.put(0.0, 0.0)
    amp_ts.put(t_ramp, amp_max)
    amp_ts.put(t_sweep - t_ramp, amp_max)
    amp_ts.put(t_sweep, 0.0)
    if t_hold > 0:
        amp_ts.put(t_end, 0.0)          # удержание без драйва

    phase_ts = TimeSeries()
    phase_ts.put(0.0, 0.0)
    phase_ts.put(t_end, 0.0)

    det_ts = TimeSeries()
    det_ts.put(0.0, det_start)
    det_ts.put(t_sweep, det_end)
    if t_hold > 0:
        det_ts.put(t_end, det_end)      # финальный detuning удерживается

    return DrivingField(amplitude=amp_ts, phase=phase_ts, detuning=det_ts)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--local"
    # ВАЖНО: для железа берём ПОВЁРНУТОЕ кольцо из phase2 (квирк Aquila: y-ряды 0 или >=4мкм).
    # Оно же использовалось для базовой точки 0.581 — сравнимость сохраняется.
    from phase2_aquila_submit import ring8
    register = ring8()

    if mode == "--local":
        sim = ex.LocalSimulator("braket_ahs")
        print("Локальная валидация (идеальный симулятор: распада БЫТЬ НЕ ДОЛЖНО)")
        for t_hold in [0.0] + HOLDS:
            drive = build_drive_with_hold(t_hold)
            prog = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)
            res = sim.run(prog, shots=100).result()
            ret = ex.compute_retention(res.measurements, "ring")
            print(f"  t_hold={t_hold*1e6:5.1f} мкс → retention {ret:.3f}")
        print("Если тут плоско — программа валидна; распад на ЖЕЛЕЗЕ будет настоящим эффектом.")
        return

    if mode in ("--submit", "--submit2"):
        from braket.aws import AwsDevice
        holds, shots, tag = (HOLDS, SHOTS, "") if mode == "--submit" else (HOLDS2, SHOTS2, "b")
        globals()["HOLDS"], globals()["SHOTS"] = holds, shots
        est = len(holds) * (PER_TASK + shots * PER_SHOT)
        assert est <= LIMIT_REMAINING, f"смета ${est:.2f} > остатка лимита ${LIMIT_REMAINING}"
        print(f"Смета: ${est:.2f} (остаток лимита ${LIMIT_REMAINING}) — ок")
        dev = AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
        print("Статус:", dev.status)
        tasks = []
        for t_hold in holds:
            drive = build_drive_with_hold(t_hold)
            prog = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)
            prog = prog.discretize(dev)
            task = dev.run(prog, shots=shots)
            meta = {"name": f"ring8-hold{t_hold*1e6:g}us", "t_hold_s": t_hold,
                    "arn": task.id, "shots": shots,
                    "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            print(f"  {meta['name']}: {task.id}")
            tasks.append(meta)
        with open(f"h23-tasks{tag}.json", "w") as f:
            json.dump(tasks, f, indent=1, ensure_ascii=False)
        print(f"ARN-ы сохранены в h23-tasks{tag}.json")


if __name__ == "__main__":
    main()
