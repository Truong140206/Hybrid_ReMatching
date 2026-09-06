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

Log files are addressed by their EXACT name, rebuilt from the same template
eval_rp_head_any_4090.sh uses, not by a loose glob. The first version of this
tool globbed for the fusion weights alone and took whichever file sorted first,
which picked up a stray log from an old sweep: it reported 75.08 for the full
method on Sup-21K where the measured value is 75.51. Every hyperparameter that
appears in the name is therefore pinned here, and a cell whose exact file is
absent is reported missing rather than filled from a near neighbour.

A_N in their table is average final accuracy, which is our Acc@1: their
published Sup-21K figure of 73.86 sits inside our reproduction's 73.94 +/- 0.48.
So Acc@1 is what this prints, with Acc@task available via --metric.

Usage:
    python tools/ablation_table.py
    python tools/ablation_table.py --metric Acc@task --show-files
"""
import argparse
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET, NUM_TASKS = 'imr', 10

# (directory tag, tag inside the log name, label). The two tags differ: the run
# directory says `ibot1k` while the log name says `bibot`, because one is our
# naming and the other is derived from the timm model name by the eval script.
BACKBONES = [
    ('', '', 'Sup-21K'),
    ('mocov3', 'bmocov3', 'MoCo-1K'),
    ('ibot1k', 'bibot', 'iBOT-1K'),
    ('ibot21k', 'b21kibot', 'iBOT-21K'),
    ('dino', 'bdino', 'DINO-1K'),
    ('mae', 'bmae', 'MAE-1K'),
]

# (label, fusion weight tag, class weight tag, gate tag). With beta = 0 the gate
# has nothing to weight, so the first two rows carry `gmargin` only because that
# is how they were run; it changes nothing there.
ARMS = [
    ('Moc (w=1, b=0)', 'w1p0', 'cw0p0', 'gmargin'),
    ('+ tang dinh tuyen', 'w0p7', 'cw0p0', 'gmargin'),
    ('+ tang phan lop', 'w0p7', 'cw0p5', ''),
    ('+ cong bien (day du)', 'w0p7', 'cw0p5', 'gmargin'),
]

# Everything else in the log name, fixed across this table.
FIXED = ('_eval_rp_lora_d10000_relu_l10000_nnone_t0_b0p0_p1_inone_c0_ra0ls0'
         '_f1d1')


def output_root():
    return os.path.join(os.path.dirname(REPO_ROOT), 'hrm-pet-output')


def log_path(root, dir_tag, log_tag, weight, class_weight, gate):
    suffix = '_%s' % dir_tag if dir_tag else ''
    base = '%s%s_lora_rank8_baseline_%dtasks_seed42' % (
        DATASET, suffix, NUM_TASKS)
    name = '%s%s%slsw0p0c0ca0%ssh1p0m1%s%s.log' % (
        base, FIXED, weight, class_weight, gate, log_tag)
    return os.path.join(root, name)


def final_value(path, metric):
    if not os.path.isfile(path):
        return None
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--metric', default='Acc@1',
                        help='Acc@1 (nhu A_N cua ho) hoac Acc@task')
    parser.add_argument('--show-files', action='store_true',
                        help='in ten tep da doc cho tung o')
    args = parser.parse_args()
    root = args.root or output_root()

    print('Ablation cong don tren Split-ImageNet-R, seed 42, chi so %s'
          % args.metric)
    print('(hinh dang Bang 3 cua bai goc: thanh phan cong don, backbone lam cot)')
    print('Tep duoc dia chi hoa chinh xac, khong doan gan dung.\n')

    labels = [label for label, _, _, _ in ARMS]
    width = max(len(label) for label in labels) + 1
    header = ' ' * width + ''.join('%10s' % label for _, _, label in BACKBONES)
    print(header)
    print('-' * len(header))

    table, missing = {}, []
    for label, weight, class_weight, gate in ARMS:
        cells = []
        for dir_tag, log_tag, name in BACKBONES:
            path = log_path(root, dir_tag, log_tag, weight, class_weight, gate)
            value = final_value(path, args.metric)
            table[(label, name)] = value
            if value is None:
                missing.append((label, name, path))
            cells.append('%10.2f' % value if value is not None
                         else '%10s' % '--')
            if args.show_files:
                print('    %-22s %-9s %s' % (label, name,
                                             os.path.basename(path)))
        if not args.show_files:
            print('%-*s%s' % (width, label, ''.join(cells)))

    if args.show_files:
        return 0

    print('\nMuc dong gop cua tung thanh phan (hieu voi hang tren):')
    print(header)
    print('-' * len(header))
    for i in range(1, len(ARMS)):
        label = labels[i]
        cells = []
        for _, _, name in BACKBONES:
            here, above = table[(label, name)], table[(labels[i - 1], name)]
            cells.append('%+10.2f' % (here - above)
                         if here is not None and above is not None
                         else '%10s' % '--')
        print('%-*s%s' % (width, label, ''.join(cells)))

    if missing:
        print('\nThieu %d o (khong tim thay dung tep):' % len(missing))
        for label, name, path in missing:
            print('  %-22s %-9s %s' % (label, name, os.path.basename(path)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
