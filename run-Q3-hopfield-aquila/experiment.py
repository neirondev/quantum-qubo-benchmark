#!/usr/bin/env python3
"""
experiment.py — Phase 1: Local AHS degradation curves for Aquila (braket_ahs).
Three geometries, three parameter sweeps, retention metric.
"""

import json
import math
import os
import sys

import numpy as np

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern
from braket.devices import LocalSimulator
from braket.timings.time_series import TimeSeries

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
RABI_MAX = 15.7e6           # rad/s, trapezoid plateau
DET_START = -1.2e8          # rad/s, initial detuning
DET_END_NOMINAL = 1.2e8     # rad/s, nominal final detuning
T_SWEEP_NOMINAL = 4e-6      # s, nominal sweep duration
T_RAMP = 0.1e-6             # s, trapezoid ramp-up / ramp-down time

# Device validation limits (SDK enforces these)
FIELD_X_MAX = 75e-6         # m
FIELD_Y_MAX = 76e-6         # m
MIN_SPACING = 4e-6          # m
DET_LIMIT = 1.25e8          # rad/s, absolute
RABI_LIMIT = 15.8e6         # rad/s

# ─────────────────────────────────────────────
#  Geometry builders
# ─────────────────────────────────────────────

def build_line(n=8, step=6e-6):
    """Line of n atoms with uniform spacing `step` (metres)."""
    reg = AtomArrangement()
    for i in range(n):
        reg.add((i * step, 0.0))
    return reg


def build_ring(n=8, step_arc=6e-6):
    """Ring of n atoms; chord length between neighbours ≈ step_arc."""
    circumference = n * step_arc
    radius = circumference / (2.0 * math.pi)
    reg = AtomArrangement()
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        reg.add((radius * math.cos(theta), radius * math.sin(theta)))
    return reg


def build_two_clusters(cluster_size=4, intra_step=6e-6, inter_gap=25e-6):
    """Two 4-atom rows (clusters) with gap `inter_gap` between them."""
    reg = AtomArrangement()
    # cluster A at y = 0
    for i in range(cluster_size):
        reg.add((i * intra_step, 0.0))
    # cluster B at y = inter_gap
    for i in range(cluster_size):
        reg.add((i * intra_step, inter_gap))
    return reg


# ─────────────────────────────────────────────
#  Protocol builders
# ─────────────────────────────────────────────

def build_driving_field(amp_max=RABI_MAX,
                         det_start=DET_START,
                         det_end=DET_END_NOMINAL,
                         t_sweep=T_SWEEP_NOMINAL,
                         t_ramp=T_RAMP):
    """
    Return a DrivingField with:
      - trapezoid amplitude (0 → amp_max → hold → 0)
      - linear detuning sweep (det_start → det_end)
      - zero phase
    """
    # amplitude
    amp_ts = TimeSeries()
    amp_ts.put(0.0, 0.0)
    amp_ts.put(t_ramp, amp_max)
    amp_ts.put(t_sweep - t_ramp, amp_max)
    amp_ts.put(t_sweep, 0.0)

    # phase (constant 0)
    phase_ts = TimeSeries()
    phase_ts.put(0.0, 0.0)
    phase_ts.put(t_sweep, 0.0)

    # detuning (linear sweep)
    det_ts = TimeSeries()
    det_ts.put(0.0, det_start)
    det_ts.put(t_sweep, det_end)

    return DrivingField(
        amplitude=Field(amp_ts, Pattern([])),
        phase=Field(phase_ts, Pattern([])),
        detuning=Field(det_ts, Pattern([])),
    )


def build_noisy_driving_field(amp_max=RABI_MAX,
                               det_start=DET_START,
                               det_end=DET_END_NOMINAL,
                               t_sweep=T_SWEEP_NOMINAL,
                               t_ramp=T_RAMP,
                               sigma_amp=0.0):
    """
    Like build_driving_field but the amplitude plateau value is jittered
    by multiplicative Gaussian noise with std `sigma_amp`.
    The jitter is applied to the hold value only (ramp edges use the same
    jittered amplitude to keep the trapezoid shape).
    """
    if sigma_amp <= 0.0:
        return build_driving_field(amp_max, det_start, det_end, t_sweep, t_ramp)

    # Jitter the plateau amplitude
    noise = np.random.normal(0.0, sigma_amp * amp_max)
    amp_hold = max(0.0, amp_max + noise)

    amp_ts = TimeSeries()
    amp_ts.put(0.0, 0.0)
    amp_ts.put(t_ramp, amp_hold)
    amp_ts.put(t_sweep - t_ramp, amp_hold)
    amp_ts.put(t_sweep, 0.0)

    phase_ts = TimeSeries()
    phase_ts.put(0.0, 0.0)
    phase_ts.put(t_sweep, 0.0)

    det_ts = TimeSeries()
    det_ts.put(0.0, det_start)
    det_ts.put(t_sweep, det_end)

    return DrivingField(
        amplitude=Field(amp_ts, Pattern([])),
        phase=Field(phase_ts, Pattern([])),
        detuning=Field(det_ts, Pattern([])),
    )


# ─────────────────────────────────────────────
#  Target patterns
# ─────────────────────────────────────────────
#  post_sequence:  1 = ground,  0 = Rydberg

def _target_patterns_line_ring():
    """Antiferromagnetic order – two mirror variants."""
    n = 8
    return [
        tuple([i % 2 for i in range(n)]),          # 0,1,0,1,...
        tuple([(i + 1) % 2 for i in range(n)]),    # 1,0,1,0,...
    ]


def _target_patterns_clusters():
    """AF order within each 4-atom cluster independently."""
    patterns = []
    for a_block in [(0, 1, 0, 1), (1, 0, 1, 0)]:
        for b_block in [(0, 1, 0, 1), (1, 0, 1, 0)]:
            patterns.append(tuple(list(a_block) + list(b_block)))
    return patterns


TARGET_PATTERNS = {
    'line': _target_patterns_line_ring(),
    'ring': _target_patterns_line_ring(),
    'clusters': _target_patterns_clusters(),
}


def is_target(post_seq, geo_name):
    return tuple(post_seq) in TARGET_PATTERNS[geo_name]


# ─────────────────────────────────────────────
#  Retention
# ─────────────────────────────────────────────

def compute_retention(measurements, geo_name):
    total = len(measurements)
    if total == 0:
        return 0.0
    good = sum(1 for shot in measurements
               if shot.status.name == 'SUCCESS' and is_target(shot.post_sequence, geo_name))
    return good / total


# ─────────────────────────────────────────────
#  Sweep definitions & single-point runners
# ─────────────────────────────────────────────

SWEEPS = [
    {
        'param': 'final_detuning',
        'values': [1.2e8, 0.8e8, 0.4e8, 0.0, -0.4e8, -0.8e8, -1.2e8],
        'desc': 'Final detuning (rad/s)',
    },
    {
        'param': 't_sweep',
        'values': [4e-6, 3e-6, 2e-6, 1.5e-6, 1e-6, 0.5e-6],
        'desc': 'Sweep duration (s)',
    },
    {
        'param': 'sigma_amp',
        'values': [0.0, 0.01, 0.03, 0.05, 0.1],
        'desc': 'Amplitude jitter σ (multiplicative)',
    },
]

GEOMETRIES = {
    'line': build_line,
    'ring': build_ring,
    'clusters': build_two_clusters,
}

RESULTS_FILE = 'results-phase1.json'
N_SHOTS = 100


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def run_point_batch(sim, register, geo_name, param_name, param_value, n_shots=N_SHOTS):
    """Run one parameter point (non-noise case) with a single batch."""
    if param_name == 'final_detuning':
        drive = build_driving_field(det_end=param_value)
    elif param_name == 't_sweep':
        drive = build_driving_field(t_sweep=param_value)
    else:
        raise ValueError(f"Unexpected param for batch run: {param_name}")

    ah_sim = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)
    task = sim.run(ah_sim, shots=n_shots)
    result = task.result()
    return compute_retention(result.measurements, geo_name)


def run_point_noise(sim, register, geo_name, sigma, n_shots=N_SHOTS):
    """Run amplitude-noise point: generate ONE noisy amplitude and run all shots."""
    drive = build_noisy_driving_field(sigma_amp=sigma)
    ah_sim = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)
    task = sim.run(ah_sim, shots=n_shots)
    result = task.result()
    return compute_retention(result.measurements, geo_name)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    # Seed for reproducibility
    np.random.seed(42)

    sim = LocalSimulator('braket_ahs')
    results = load_results()
    existing = {(r['geometry'], r['param'], r['value'])
                for r in results}

    total_points = (len(GEOMETRIES) *
                    sum(len(sweep['values']) for sweep in SWEEPS))
    completed = 0

    print(f"LocalSimulator backend: {sim.name}")
    print(f"Total experiment points: {total_points}")
    print(f"Already completed: {len(results)}")

    for geo_name, build_fn in GEOMETRIES.items():
        print(f"\n{'=' * 60}")
        print(f"  Geometry: {geo_name}")
        print(f"{'=' * 60}")
        register = build_fn()

        for sweep in SWEEPS:
            param_name = sweep['param']
            print(f"\n  --- {sweep['desc']} ---")

            for value in sweep['values']:
                key = (geo_name, param_name, value)
                if key in existing:
                    print(f"    SKIP  {param_name}={value}  (already done)")
                    completed += 1
                    continue

                print(f"    RUN   {param_name}={value}  ...  ", end='', flush=True)
                try:
                    if param_name == 'sigma_amp':
                        ret = run_point_noise(sim, register, geo_name, value)
                    else:
                        ret = run_point_batch(sim, register, geo_name,
                                              param_name, value)

                    entry = {
                        'geometry': geo_name,
                        'param': param_name,
                        'value': value,
                        'shots': N_SHOTS,
                        'retention': round(ret, 6),
                    }
                    results.append(entry)
                    save_results(results)
                    print(f"retention={ret:.4f}")
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    save_results(results)

                completed += 1
                print(f"    Progress: {completed}/{total_points}")

    print(f"\n{'=' * 60}")
    print(f"  DONE.  Total results saved: {len(results)}")
    print(f"  File:  {RESULTS_FILE}")


if __name__ == '__main__':
    main()