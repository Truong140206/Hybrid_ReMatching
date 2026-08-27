#!/usr/bin/env python3
"""What can be evaluated without training anything.

The two fusion stages live entirely at inference time, so widening the reported
scope does not always mean retraining. It means finding LoRA and TII checkpoints
that already exist for a configuration we have not reported yet -- a different
LoRA rank, a different task count, a dataset trained for some earlier ablation
and then abandoned.

The evaluation path is already agnostic to rank and task count:
eval_rp_head_any_4090.sh reads the rank out of the checkpoint rather than taking
it as an argument, and NUM_TASKS is an environment variable. So any complete
(run, TII) pair is one evaluation away from being a reportable data point.

A configuration is USABLE when three things are present:
  - the LoRA run directory with every task checkpoint
  - the matching TII directory with every task checkpoint
  - the conventional baseline log, without which there is nothing to compare to

Usage: python tools/inventory_checkpoints.py [OUTPUT_ROOT]
"""
import os
import re
import sys
from collections import defaultdict

RUN = re.compile(r'^(?P<prefix>[a-z0-9]+)_lora_rank(?P<rank>\d+)_'
                 r'(?P<tag>.+?)_(?P<tasks>\d+)tasks_seed(?P<seed>\d+)$')

REPORTED = {('imr', 8, 10), ('cifar100', 8, 10), ('cub200', 8, 10)}


def complete(run_dir, tasks):
    """Every task checkpoint present and non-empty."""
    for task in range(1, tasks + 1):
        path = os.path.join(run_dir, 'checkpoint',
                            'task%d_checkpoint.pth' % task)
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            return False
    return True


def main():
    root = (sys.argv[1] if len(sys.argv) > 1 else
            os.path.expanduser('~/Documents/truongnguyen/hrm-pet-output'))
    if not os.path.isdir(root):
        print('Khong co thu muc: %s' % root)
        return 2
    print('OUTPUT_ROOT: %s\n' % root)

    groups = defaultdict(list)
    for name in sorted(os.listdir(root)):
        found = RUN.match(name)
        if not found or not os.path.isdir(os.path.join(root, name)):
            continue
        info = found.groupdict()
        prefix, rank = info['prefix'], int(info['rank'])
        tasks, seed = int(info['tasks']), int(info['seed'])

        run_dir = os.path.join(root, name)
        tii_dir = os.path.join(root, '%s_tii_original_%dtasks_seed%d'
                               % (prefix, tasks, seed))
        conv = os.path.join(root, name + '_eval_conventional.log')

        groups[(prefix, rank, tasks)].append({
            'seed': seed, 'tag': info['tag'],
            'lora': complete(run_dir, tasks),
            'tii': os.path.isdir(tii_dir) and complete(tii_dir, tasks),
            'conv': os.path.isfile(conv) and os.path.getsize(conv) > 0,
        })

    if not groups:
        print('Khong tim thay thu muc chay nao khop mau.')
        return 1

    print('%-10s %5s %6s  %-28s %s' % ('Bo', 'Hang', 'Tacvu', 'Seed dung duoc',
                                       'Trang thai'))
    print('-' * 78)
    fresh = []
    for key in sorted(groups):
        prefix, rank, tasks = key
        rows = groups[key]
        ok = sorted(r['seed'] for r in rows
                    if r['lora'] and r['tii'] and r['conv'])
        partial = sorted(r['seed'] for r in rows
                         if (r['lora'] and r['tii']) and not r['conv'])
        broken = sorted(r['seed'] for r in rows if not (r['lora'] and r['tii']))

        if key in REPORTED:
            status = 'DA BAO CAO'
        elif ok:
            status = '*** DUNG DUOC NGAY, chua bao cao ***'
            fresh.append((key, ok))
        elif partial:
            status = 'thieu log baseline thuong'
        else:
            status = 'checkpoint chua du'
        detail = ','.join(str(s) for s in ok) or '-'
        if partial:
            detail += '  (thieu baseline: %s)' % ','.join(
                str(s) for s in partial)
        if broken:
            detail += '  (hong: %s)' % ','.join(str(s) for s in broken)
        print('%-10s %5d %6d  %-28s %s' % (prefix, rank, tasks, detail, status))

    print()
    if fresh:
        print('Mo rong duoc ma KHONG can huan luyen:')
        for (prefix, rank, tasks), seeds in fresh:
            print('  %s, hang %d, %d tac vu -- seed %s'
                  % (prefix, rank, tasks, ','.join(str(s) for s in seeds)))
        print('\nChay bang eval_rp_head_any_4090.sh voi NUM_TASKS tuong ung;')
        print('hang tu doc ra tu checkpoint nen khong can dat gi them.')
    else:
        print('Khong co cau hinh nao san sang ngoai ba cau hinh da bao cao.')
        print('Moi truc mo rong con lai deu can huan luyen lai LoRA va TII.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
