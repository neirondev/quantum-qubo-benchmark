#!/usr/bin/env python3
"""Анализ измерений Aquila: топ-паттерны, расстояния Хэмминга до цели,
проверка модели независимого пер-атомного шума retention ≈ f^N."""
import json
from collections import Counter

from braket.aws import AwsQuantumTask

TARGETS = {
    'ring8-nominal': [(0, 1, 0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0, 1, 0)],
    'line7-det1.1e8': [(0, 1, 0, 1, 0, 1, 0)],
}

tasks = json.load(open('phase2-tasks.json'))
for t in tasks:
    task = AwsQuantumTask(arn=t['arn'])
    ms = task.result().measurements
    good = [tuple(m.post_sequence) for m in ms if all(b == 1 for b in m.pre_sequence)]
    tgts = TARGETS[t['name']]
    n = len(tgts[0])
    hd = Counter(min(sum(a != b for a, b in zip(s, tg)) for tg in tgts) for s in good)
    hit = hd.get(0, 0)
    f = (hit / len(good)) ** (1 / n)
    print(f"== {t['name']}: {len(good)} загруженных шотов, цель {hit} ({hit/len(good):.3f})")
    print('   Хэмминг до цели:', dict(sorted(hd.items())))
    print(f'   пер-атомная верность f = retention^(1/{n}) = {f:.3f}')
    top = Counter(good).most_common(5)
    print('   топ-5:', ', '.join(f"{''.join(map(str, p))}x{c}" for p, c in top))
