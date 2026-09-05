#!/usr/bin/env python3
"""Collect the baseline-vs-proposed grid straight out of the evaluation logs.

Twenty cells were measured across four benchmarks and five backbones, over
several days and several restarts, with some logs written by runs that later
crashed. Transcribing those numbers by hand into a report is exactly the sort
of step that quietly introduces an error nobody can trace afterwards, so the
table is built from the files instead.

Each evaluation log is named after the run directory plus the fusion settings,
so the dataset, backbone and configuration are all recoverable from the path:

    ima_ibot1k_lora_rank8_baseline_10tasks_seed42_eval_rp_lora_..._f1d1w1p0...cw0p0...

  w1p0 with cw0p0  -> baseline   (w = 1.0, beta = 0, routing goes to TII alone)
  w0p7 with cw0p5  -> proposed   (w = 0.7, beta = 0.5)

A log with no final "Average accuracy till taskN" row is one of the crashed
runs and is reported as missing rather than skipped silently.

Usage:
    python tools/collect_results.py
    python tools/collect_results.py --csv results.csv
"""
import argparse
import glob
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = [('imr', 'ImageNet-R', 10), ('cifar100', 'CIFAR-100', 10),
            ('ima', 'ImageNet-A', 10), ('fivedatasets', '5-Datasets', 5)]
# Several aliases per backbone: ImageNet-R was trained before the naming
# convention settled, so MoCo v3 sits under `mocov3` there and `moco1k`
# elsewhere. Renaming directories that hold measured results would break the
# link between a number and the run that produced it, so both names are read.
BACKBONES = [(('',), 'Sup-21K'), (('moco1k', 'mocov3'), 'MoCo v3-1K'),
             (('ibot1k',), 'iBOT-1K'), (('ibot21k',), 'iBOT-21K'),
             (('dino',), 'DINO-1K'), (('mae',), 'MAE-1K')]
METRICS = ['Acc@task', 'Acc@1', 'Acc@5', 'Loss', 'Forgetting', 'Backward']
# +1 where a larger number is better, -1 where a smaller one is. Backward is
# normally negative and closer to zero is better, so it counts as larger.
DIRECTION = {'Acc@task': 1, 'Acc@1': 1, 'Acc@5': 1,
             'Loss': -1, 'Forgetting': -1, 'Backward': 1}

# The two configurations the grid compares, as they appear in the log name.
ARMS = {'baseline': ('w1p0', 'cw0p0'), 'proposed': ('w0p7', 'cw0p5')}


def output_root():
    return os.path.join(os.path.dirname(REPO_ROOT), 'hrm-pet-output')


def final_row(path, num_tasks):
    marker = 'Average accuracy till task%d]' % num_tasks
    row = None
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if marker in line:
                row = line
    if row is None:
        return None
    out = {}
    for name in METRICS:
        match = re.search(r'%s:\s*(-?[0-9]+(?:\.[0-9]+)?)' % re.escape(name), row)
        if match:
            out[name] = float(match.group(1))
    return out or None


def find_log(root, dataset, tags, num_tasks, arm):
    weight, class_weight = ARMS[arm]
    hits = []
    for tag in tags:
        suffix = '_%s' % tag if tag else ''
        base = '%s%s_lora_rank8_baseline_%dtasks_seed42' % (
            dataset, suffix, num_tasks)
        for path in glob.glob(os.path.join(root, base + '_eval_rp_*.log')):
            name = os.path.basename(path)
            if ('f1d1' + weight) in name and ('cw' + class_weight[2:]) in name:
                hits.append(path)
    if not hits:
        return None
    # Prefer the newest, in case an older naming variant lingers.
    return max(hits, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--csv', default=None)
    parser.add_argument('--worse', action='store_true',
                        help='liet ke moi o va chi so ma de xuat kem hon moc')
    args = parser.parse_args()
    root = args.root or output_root()

    rows = []
    print('%-12s %-11s %9s %9s %8s   %9s %9s %8s   %7s %7s'
          % ('bo du lieu', 'backbone', 'moc@task', 'dx@task', 'delta',
             'moc@1', 'dx@1', 'delta', 'quen-m', 'quen-d'))
    print('-' * 104)
    for dataset, dataset_label, num_tasks in DATASETS:
        for tags, backbone_label in BACKBONES:
            found = {}
            for arm in ARMS:
                path = find_log(root, dataset, tags, num_tasks, arm)
                found[arm] = final_row(path, num_tasks) if path else None
            if not any(found.values()):
                continue
            base, prop = found['baseline'], found['proposed']
            if base is None or prop is None:
                which = 'moc' if base is None else 'de xuat'
                print('%-12s %-11s  THIEU %s' % (dataset_label, backbone_label, which))
                continue
            print('%-12s %-11s %9.2f %9.2f %+8.2f   %9.2f %9.2f %+8.2f   %7.2f %7.2f'
                  % (dataset_label, backbone_label,
                     base['Acc@task'], prop['Acc@task'],
                     prop['Acc@task'] - base['Acc@task'],
                     base['Acc@1'], prop['Acc@1'], prop['Acc@1'] - base['Acc@1'],
                     base['Forgetting'], prop['Forgetting']))
            row = {'dataset': dataset_label, 'backbone': backbone_label}
            for name in METRICS:
                row['base_' + name] = base.get(name)
                row['prop_' + name] = prop.get(name)
            rows.append(row)
        print()

    print('tong: %d o co du ca hai cau hinh' % len(rows))

    if args.worse:
        # Sweeping only the metrics the table prints would answer the
        # question from partial data, so this covers all six.
        print()
        print('Moi cho de xuat KEM hon moc, tren ca sau chi so:')
        found = 0
        for row in rows:
            for name in METRICS:
                base, prop = row['base_' + name], row['prop_' + name]
                if base is None or prop is None:
                    continue
                delta = (prop - base) * DIRECTION[name]
                if delta < 0:
                    found += 1
                    print('  %-12s %-11s %-11s %9.4f -> %9.4f  (kem %.4f)'
                          % (row['dataset'], row['backbone'], name,
                             base, prop, -delta))
        if not found:
            print('  khong co cho nao')
        print('  -> %d cho kem hon tren %d o x %d chi so = %d phep so sanh'
              % (found, len(rows), len(METRICS), len(rows) * len(METRICS)))

    if args.csv and rows:
        import csv
        with open(args.csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('da ghi %s' % args.csv)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
