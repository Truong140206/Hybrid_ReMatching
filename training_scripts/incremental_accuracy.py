#!/usr/bin/env python3
"""Per-stage accuracy and average incremental accuracy for one or more logs.

The final row of a run hides what happened on the way there. A change that
leaves the final state untouched can still move Forgetting and Backward, since
both score a task against its own earlier peak -- so read the whole curve
before believing either metric.

Usage: python3 training_scripts/incremental_accuracy.py LOG [LOG...]
"""
import os
import re
import sys

ROW = re.compile(r'Average accuracy till task(\d+)\]\s*'
                 r'Acc@task:\s*(-?[\d.]+)\s*Acc@1:\s*(-?[\d.]+)')


def curve(path):
    stages = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            m = ROW.search(line)
            if m:
                stages[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return [stages[k] for k in sorted(stages)]


def label(path):
    name = os.path.basename(path)
    for cut in ('_eval_conventional.log', '.log'):
        if name.endswith(cut):
            name = name[:-len(cut)]
    name = re.sub(r'^(imr|cifar100|cub200)_lora_rank8_baseline_10tasks_', '', name)
    # Keep beta as well as the gate. The old pattern captured only the gate tag,
    # so a beta sweep -- the commonest thing this script is pointed at -- came
    # out as several identical 'gmargin' labels with no way to tell the columns
    # apart. It also failed to match 'gmargin_both' at all, since that tag does
    # not end where the pattern expected.
    m = re.search(r'cw(\d+p\d+)sh\d+p\d+m\d+(g[a-z_]+)?(r\d+p\d+)?$', name)
    if m:
        gate = (m.group(2) or 'gnone')[1:]
        ramp = ('/' + m.group(3)) if m.group(3) else ''
        return '%s b%s%s' % (gate, m.group(1).replace('p', '.'), ramp)
    return name[-28:] or name


for path in sys.argv[1:]:
    rows = curve(path)
    if not rows:
        print('%-20s no stage rows in %s' % ('--', os.path.basename(path)))
        continue
    acc1 = [r[1] for r in rows]
    acct = [r[0] for r in rows]
    print('%-20s AIA@1 %6.2f  AIA@task %6.2f  |  %s' % (
        label(path), sum(acc1) / len(acc1), sum(acct) / len(acct),
        ' '.join('%.1f' % a for a in acc1)))
