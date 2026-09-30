#!/usr/bin/env python3
"""Q3 фаза 2: проверка статуса и съём результатов задач Aquila.
Запуск: python phase2_fetch.py            — статусы
        python phase2_fetch.py --collect  — если COMPLETED, посчитать retention
                                            той же метрикой, что фаза 1, и записать
                                            phase2-results.json
"""
import json
import sys

from braket.aws import AwsQuantumTask

import experiment as ex  # compute_retention — ЕДИНАЯ метрика с фазой 1

TARGET_GEO = {'ring8-nominal': 'ring', 'line7-det1.1e8': 'line7'}


def retention_line7(measurements):
    """P7: единственное целевое MIS — Rydberg на позициях 1,3,5,7 (post 0=Rydberg)."""
    tgt = (0, 1, 0, 1, 0, 1, 0)
    seqs = [tuple(m.post_sequence) for m in measurements]
    ok = sum(1 for s in seqs if s == tgt)
    return ok / len(seqs)


def main():
    tasks = json.load(open('phase2-tasks.json'))
    collect = '--collect' in sys.argv
    out = []
    for t in tasks:
        task = AwsQuantumTask(arn=t['arn'])
        state = task.state()
        print(f"{t['name']}: {state}")
        if collect and state == 'COMPLETED':
            res = task.result()
            ms = res.measurements
            # успешные шоты: pre_sequence все 1 (атомы загружены)
            good = [m for m in ms if all(b == 1 for b in m.pre_sequence)]
            geo = TARGET_GEO[t['name']]
            if geo == 'ring':
                ret = ex.compute_retention(good, 'ring')
            else:
                ret = retention_line7(good)
            rec = {**t, 'state': state, 'shots_total': len(ms),
                   'shots_loaded': len(good), 'retention': ret}
            print(f"  загрузка атомов: {len(good)}/{len(ms)}; retention = {ret:.3f}")
            out.append(rec)
        elif collect:
            out.append({**t, 'state': state})
    if collect:
        with open('phase2-results.json', 'w') as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print('записано phase2-results.json')


if __name__ == '__main__':
    main()
