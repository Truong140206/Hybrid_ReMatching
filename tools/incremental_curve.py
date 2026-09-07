#!/usr/bin/env python3
"""Compare two arms over the whole incremental curve, not just its last point.

Removing the class-fusion gate costs Forgetting on 20 of 21 cells and Backward
on all 21, which reads as a loss of stability. On two cells it is not: the gated
arm sits BELOW the ungated one at nine of ten stages and only draws level at the
last, so its smaller Forgetting comes from a lower earlier peak rather than from
better retention. Forgetting subtracts a task's final accuracy from its own
earlier peak, so lowering the peak improves it for free.

Two cells are an anecdote. This reads every stage of every log and reports the
average incremental accuracy -- the standard continual-learning summary of the
whole curve -- together with how many stages each arm actually leads, so the
claim can be checked across the grid instead of asserted from a sample.

Usage:
    python tools/incremental_curve.py --arms relative,nogate03
    python tools/incremental_curve.py --arms proposed,nogate03 --metric Acc@1
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_results as cr


def stages(path, num_tasks, metric):
    """Every stage's value, or None if any stage is missing.

    A run that died partway leaves a log with some stages present, and averaging
    over the ones that survived would silently compare a short curve against a
    full one. An incomplete log is reported missing instead.
    """
    values = {}
    pattern = re.compile(r'till task(\d+)\]')
    grab = re.compile(r'%s:\s*(-?[0-9]+(?:\.[0-9]+)?)' % re.escape(metric))
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            found = pattern.search(line)
            if not found:
                continue
            got = grab.search(line)
            if got:
                values[int(found.group(1))] = float(got.group(1))
    out = [values.get(t) for t in range(1, num_tasks + 1)]
    return None if any(v is None for v in out) else out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--metric', default='Acc@1')
    parser.add_argument('--arms', default='relative,nogate03',
                        help='hai nhanh can so, cach nhau boi dau phay: %s'
                             % ', '.join(sorted(cr.ARMS)))
    args = parser.parse_args()
    root = args.root or cr.output_root()

    try:
        left, right = [a.strip() for a in args.arms.split(',')]
    except ValueError:
        print('--arms can dung hai ten, cach nhau boi dau phay')
        return 1
    for arm in (left, right):
        if arm not in cr.ARMS:
            print('khong biet nhanh %r; co: %s'
                  % (arm, ', '.join(sorted(cr.ARMS))))
            return 1

    print('Trung binh %s tang dan tren ca duong cong, khong chi diem cuoi\n'
          % args.metric)
    print('%-12s %-11s %9s %9s %8s %9s %11s'
          % ('bo du lieu', 'backbone', left[:8], right[:8], 'chenh',
             'cuoi-chenh', 'phai dan dau'))
    print('-' * 78)

    totals = {'delta': 0.0, 'cells': 0, 'right_ahead': 0,
              'right_final_ahead': 0, 'stage_wins': 0, 'stage_total': 0}
    for dataset, dataset_label, num_tasks in cr.DATASETS:
        for dir_tags, log_tag, backbone_label in cr.BACKBONES:
            curves = {}
            for arm in (left, right):
                path = cr.find_log(root, dataset, dir_tags, log_tag, num_tasks,
                                   arm)
                curves[arm] = stages(path, num_tasks,
                                     args.metric) if path else None
            if curves[left] is None or curves[right] is None:
                if curves[left] is not None or curves[right] is not None:
                    which = left if curves[left] is None else right
                    print('%-12s %-11s  THIEU %s'
                          % (dataset_label, backbone_label, which))
                continue
            a, b = curves[left], curves[right]
            mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
            ahead = sum(1 for x, y in zip(a, b) if y > x)
            print('%-12s %-11s %9.2f %9.2f %+8.2f %+9.2f %8d/%-2d'
                  % (dataset_label, backbone_label, mean_a, mean_b,
                     mean_b - mean_a, b[-1] - a[-1], ahead, num_tasks))
            totals['delta'] += mean_b - mean_a
            totals['cells'] += 1
            totals['right_ahead'] += 1 if mean_b > mean_a else 0
            totals['right_final_ahead'] += 1 if b[-1] > a[-1] else 0
            totals['stage_wins'] += ahead
            totals['stage_total'] += num_tasks
        print()

    if not totals['cells']:
        print('khong o nao co du ca hai nhanh')
        return 1
    print('%d o. %s dan ve trung binh duong cong o %d o, ve diem cuoi o %d o.'
          % (totals['cells'], right, totals['right_ahead'],
             totals['right_final_ahead']))
    print('Chenh trung binh tren duong cong: %+.2f diem.'
          % (totals['delta'] / totals['cells']))
    print('Tren tung giai doan: %s dan %d tren %d giai doan.'
          % (right, totals['stage_wins'], totals['stage_total']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
