#!/usr/bin/env python3
"""Build the cumulative component ablation, in the shape the original uses.

Table 3 of the HRM-PET paper ablates its components cumulatively -- Baseline,
+DRM, +CRM, +DRM+CRM, +DRM+CRM+CTIRD -- on ImageNet-R alone, with the five
pre-trained backbones as the columns and A_N as the entry. Our ablations exist
but are scattered across separate tables and only on Sup-21K, which makes them
harder to compare against theirs than they need to be.

This assembles the same shape for our two stages:

    baseline          w = 1.0, beta = 0     routing is TII alone
    + routing stage   w = 0.7, beta = 0     second source enters the router
    + class stage     w = 0.7, beta = 0.5   ungated class-level fusion
    + margin gate     w = 0.7, beta = 0.5   the full method

Each row is a strict superset of the one above it, so the difference between
consecutive rows is what that component is worth on that backbone.

A_N in their table is average final accuracy, which is our Acc@1: their
published Sup-21K figure of 73.86 sits inside our reproduction's 73.94 +/- 0.48.
So Acc@1 is what this prints, with Acc@task available via --metric.

Usage:
    python tools/ablation_table.py
    python tools/ablation_table.py --metric Acc@task
"""
import argparse
import glob
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET, NUM_TASKS = 'imr', 10
BACKBONES = [(('',), 'Sup-21K'), (('mocov3', 'moco1k'), 'MoCo-1K'),
             (('ibot1k',), 'iBOT-1K'), (('ibot21k',), 'iBOT-21K'),
             (('dino',), 'DINO-1K'), (('mae',), 'MAE-1K')]

# (label, fusion weight tag, class weight tag, must the margin gate be on?)
# With beta = 0 the gate has nothing to weight, so rows one and two carry the
# gate flag only because that is how they were run; it changes nothing there.
ARMS = [
    ('Moc (w=1, b=0)', 'w1p0', 'cw0p0', True),
    ('+ tang dinh tuyen', 'w0p7', 'cw0p0', True),
    ('+ tang phan lop', 'w0p7', 'cw0p5', False),
    ('+ cong bien (day du)', 'w0p7', 'cw0p5', True),
]


def output_root():
    return os.path.join(os.path.dirname(REPO_ROOT), 'hrm-pet-output')


def final_value(path, metric):
    marker = 'Average accuracy till task%d]' % NUM_TASKS
    row = None
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if marker in line:
                row = line
    if row is None:
        return None
    match = re.search(r'%s:\s*(-?[0-9]+(?:\.[0-9]+)?)' % re.escape(metric), row)
    return float(match.group(1)) if match else None


def find_log(root, tags, weight, class_weight, want_gate):
    for tag in tags:
        suffix = '_%s' % tag if tag else ''
        base = '%s%s_lora_rank8_baseline_%dtasks_seed42' % (
            DATASET, suffix, NUM_TASKS)
        for path in sorted(glob.glob(os.path.join(root, base + '_eval_rp_*.log'))):
            name = os.path.basename(path)
            if ('f1d1' + weight) not in name:
                continue
            if ('cw' + class_weight[2:]) not in name:
                continue
            if ('gmargin' in name) != want_gate:
                continue
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--metric', default='Acc@1',
                        help='Acc@1 (nhu A_N cua ho) hoac Acc@task')
    args = parser.parse_args()
    root = args.root or output_root()

    print('Ablation cong don tren Split-ImageNet-R, seed 42, chi so %s'
          % args.metric)
    print('(dung hinh dang Bang 3 cua bai goc: thanh phan cong don, '
          'backbone lam cot)\n')

    labels = [label for label, _, _, _ in ARMS]
    width = max(len(label) for label in labels) + 1
    header = ' ' * width + ''.join('%10s' % name for _, name in BACKBONES)
    print(header)
    print('-' * len(header))

    table = {}
    for label, weight, class_weight, want_gate in ARMS:
        cells = []
        for tags, name in BACKBONES:
            path = find_log(root, tags, weight, class_weight, want_gate)
            value = final_value(path, args.metric) if path else None
            table[(label, name)] = value
            cells.append('%10.2f' % value if value is not None else '%10s' % '--')
        print('%-*s%s' % (width, label, ''.join(cells)))

    print('\nMuc dong gop cua tung thanh phan (hieu voi hang tren):')
    print(header)
    print('-' * len(header))
    for i in range(1, len(ARMS)):
        label = labels[i]
        cells = []
        for _, name in BACKBONES:
            here, above = table[(label, name)], table[(labels[i - 1], name)]
            cells.append('%+10.2f' % (here - above)
                         if here is not None and above is not None
                         else '%10s' % '--')
        print('%-*s%s' % (width, label, ''.join(cells)))

    missing = [(label, name) for (label, name), value in table.items()
               if value is None]
    if missing:
        print('\nThieu %d o:' % len(missing))
        for label, name in sorted(missing):
            print('  %-22s %s' % (label, name))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
