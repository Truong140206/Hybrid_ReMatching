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
# The shipped configuration: dim = 10000, lambda = 10000, w = 0.7, beta = 0.5,
# gate = margin, min_tasks = 1. Running with no arguments now reproduces the
# reported table exactly -- verified 2026-08-28 against the published row
# (ImageNet-R Backward -0.047 +- 0.282, the single non-positive cell of 18).
#
# Two things had to be pinned to get here. The old default ended
# '...cw0p3sh1p0m1.log', which is beta 0.3 with NO gate -- neither the beta nor
# the gate of the configuration we report. And leaving lambda to a wildcard let
# the glob swallow the whole lambda sweep: five logs matched on seed 42, and
# because '0' sorts before '_', sorted()[0] picked l100000 rather than the
# shipped l10000. That would have silently reported a run 0.43 Acc@1 worse.
#
# '_c0_' is pinned too. rollout_hybrid_4090.sh defaults CALIBRATE to 1 while the
# shipped configuration is 0, so a rollout launched without CALIBRATE=0 produces
# '_c1_' logs. Those would have matched a wildcard in that position, and if they
# were the only logs for a seed nothing would have flagged it.
FUSION_GLOB = (sys.argv[2] if len(sys.argv) > 2
               else '_eval_rp_lora_d10000_relu_l10000_*_c0_ra0ls0_f1d1w0p7*cw0p5sh1p0m1gmargin.log')


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
    used, missing, ambiguous = [], [], []
    for seed in SEEDS:
        run = template % seed
        conv = final_row(run + '_eval_conventional.log')
        matches = sorted(glob.glob(run + FUSION_GLOB))
        if len(matches) > 1:
            # Refuse rather than guess. The wildcards span lambda, dim and w,
            # so one glob really can match several distinct configurations --
            # measured: five logs matched '*cw0p5sh1p0m1gmargin.log' on seed 42,
            # spanning 0.43 Acc@1. Taking sorted()[0] would have reported one of
            # them as if it were the only candidate, and this script builds the
            # table that goes in the report.
            ambiguous.append(seed)
            print('  seed %d: %d logs match, refusing to choose:' % (
                seed, len(matches)))
            for path in matches:
                print('    ' + os.path.basename(path))
            continue
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
    if ambiguous:
        print('  AMBIGUOUS seeds %s -- narrow FUSION_GLOB and re-run; this '
              'table is incomplete' % ambiguous)
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
