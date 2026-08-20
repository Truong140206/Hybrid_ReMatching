#!/usr/bin/env python3
"""Collect conventional-vs-hybrid metrics across seeds and print report rows.

Usage: python3 training_scripts/summarize_hybrid_multiseed.py [OUTPUT_ROOT] [FUSION_GLOB]
"""
import glob
import os
import re
import statistics
import sys

OUTPUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/Documents/truongnguyen/hrm-pet-output')
DATASETS = [('imr', 'ImageNet-R'), ('cifar100', 'CIFAR-100'), ('cub200', 'CUB-200')]
SEEDS = [42, 43, 44, 45]
METRICS = ['Acc@task', 'Acc@1', 'Acc@5', 'Loss', 'Forgetting', 'Backward']
LOWER_IS_BETTER = {'Loss', 'Forgetting'}
FUSION_GLOB = (sys.argv[2] if len(sys.argv) > 2
               else '_eval_rp_lora_*f1d1w0p7*cw0p3sh1p0m1.log')


def final_row(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        rows = [l for l in handle if 'Average accuracy till task10' in l]
    return rows[-1] if rows else None


def metric(row, name):
    match = re.search(rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
    return None if match is None else float(match.group(1))


def stat(values):
    if not values:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


for prefix, label in DATASETS:
    template = os.path.join(OUTPUT_ROOT, prefix + '_lora_rank8_baseline_10tasks_seed%d')
    baseline = {m: [] for m in METRICS}
    hybrid = {m: [] for m in METRICS}
    used, missing = [], []
    for seed in SEEDS:
        run = template % seed
        conv = final_row(run + '_eval_conventional.log')
        matches = sorted(glob.glob(run + FUSION_GLOB))
        fuse = final_row(matches[0] if matches else None)
        if conv is None or fuse is None:
            missing.append('%d(%s%s)' % (seed, '' if conv else 'conv ',
                                         '' if fuse else 'fusion'))
            continue
        used.append(seed)
        for name in METRICS:
            a, b = metric(conv, name), metric(fuse, name)
            if a is not None and b is not None:
                baseline[name].append(a)
                hybrid[name].append(b)
    print('\n=== %s - seeds %s (%d/%d) ===' % (
        label, used or 'NONE', len(used), len(SEEDS)))
    if missing:
        print('  missing: ' + ', '.join(missing))
    if not used:
        continue
    print('  %-11s %17s %17s %17s' % ('Metric', 'HRM-PET', 'Hybrid', 'Delta'))
    for name in METRICS:
        b, h = stat(baseline[name]), stat(hybrid[name])
        if b is None or h is None:
            continue
        diffs = [x - y for x, y in zip(hybrid[name], baseline[name])]
        d = stat(diffs)
        better = (d[0] < 0) if name in LOWER_IS_BETTER else (d[0] > 0)
        print('  %-11s %9.2f+-%-6.2f %9.2f+-%-6.2f %+9.3f+-%-6.3f%s' % (
            name, b[0], b[1], h[0], h[1], d[0], d[1],
            '' if better else '   <-- regress'))
