#!/usr/bin/env python3
"""Build the continual dataloaders for each benchmark without training.

The 5-Datasets branch of build_continual_dataloader has never been executed in
this repository -- there was no script for it, and its config was missing every
flag the evaluation sends. ImageNet-A had not been executed either, and turned
up three separate defects in a row: a split() method that does not exist, a
dataset name whose casing does not match, and an unstratified split that leaves
classes empty. Assuming the fourth attempt is clean would be optimism, not
evidence.

So this builds the real dataloaders through the real code path -- the same
build_continual_dataloader the trainers call -- and reports per task how many
training and validation samples came out, and which class ids the task owns.
It needs no GPU and trains nothing. A few minutes here is worth days there.

What to look for in the output:
  * every task has a non-zero train and val count;
  * class ids partition the whole label space with no gaps or overlaps;
  * the totals match what tools/prepare_datasets.py reported.

Usage:
    python tools/check_dataloaders.py
    python tools/check_dataloaders.py --only 5-datasets
"""
import argparse
import importlib
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# (label, config module, --dataset value, num_tasks)
BENCHMARKS = [
    ('ImageNet-R', 'imr_lora', 'Split-Imagenet-R', 10),
    ('CIFAR-100', 'cifar100_lora', 'Split-CIFAR100', 10),
    ('ImageNet-A', 'ima_lora', 'Split-Imagenet-A', 10),
    ('5-datasets', 'five_datasets_lora', '5-datasets', 5),
]


def default_data_path():
    return os.path.join(os.path.dirname(REPO_ROOT), 'datasets')


def build_args(config_name, dataset, num_tasks, data_path):
    module = importlib.import_module('configs.%s' % config_name)
    parser = argparse.ArgumentParser(add_help=False)
    module.get_args_parser(parser)
    args = parser.parse_args([])

    args.dataset = dataset
    args.num_tasks = num_tasks
    args.data_path = data_path
    args.num_workers = 0          # keep the failure mode readable
    args.pin_mem = False
    args.batch_size = 8

    # Attributes the data path reads that argparse does not define, because
    # main.py normally sets them while bringing up distributed training.
    for name, value in (('distributed', False), ('world_size', 1), ('rank', 0),
                        ('gpu', 0), ('shuffle', False)):
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def check(label, config_name, dataset, num_tasks, data_path):
    print('=' * 66)
    print('%s   (config %s, --dataset %s, %d tac vu)'
          % (label, config_name, dataset, num_tasks))
    from datasets import build_continual_dataloader

    args = build_args(config_name, dataset, num_tasks, data_path)
    loaders, _per_cls, class_mask, target_task_map = build_continual_dataloader(args)

    print('  nb_classes = %s, so tac vu = %d' % (args.nb_classes, len(loaders)))
    if len(loaders) != num_tasks:
        print('  HONG: mong doi %d tac vu' % num_tasks)
        return False

    seen = set()
    ok = True
    for i, pair in enumerate(loaders):
        n_train = len(pair['train'].dataset)
        n_val = len(pair['val'].dataset)
        ids = class_mask[i] if class_mask else []
        overlap = seen & set(ids)
        seen |= set(ids)
        note = ''
        if n_train == 0 or n_val == 0:
            note, ok = '   <-- HONG: co nua rong', False
        if overlap:
            note, ok = '%s   <-- HONG: lop trung %s' % (note, sorted(overlap)[:5]), False
        print('  tac vu %-2d %6d train, %6d val, %3d lop [%s..%s]%s'
              % (i + 1, n_train, n_val, len(ids),
                 min(ids) if ids else '-', max(ids) if ids else '-', note))

    if args.nb_classes and seen != set(range(args.nb_classes)):
        missing = sorted(set(range(args.nb_classes)) - seen)
        print('  HONG: %d lop khong thuoc tac vu nao: %s'
              % (len(missing), missing[:10]))
        ok = False
    if len(target_task_map) != args.nb_classes:
        print('  HONG: target_task_map co %d muc, cho %d'
              % (len(target_task_map), args.nb_classes))
        ok = False

    print('  -> %s' % ('OK' if ok else 'CO VAN DE'))
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default=None)
    parser.add_argument('--only', default=None,
                        help='chi kiem mot bo, vi du 5-datasets')
    args = parser.parse_args()

    data_path = args.data_path or default_data_path()
    print('thu muc du lieu: %s\n' % data_path)

    results = []
    for label, config_name, dataset, num_tasks in BENCHMARKS:
        if args.only and args.only.lower() not in (label.lower(), dataset.lower()):
            continue
        try:
            results.append((label, check(label, config_name, dataset,
                                         num_tasks, data_path)))
        except Exception:                                  # noqa: BLE001
            traceback.print_exc()
            print('  -> NEM NGOAI LE')
            results.append((label, False))
        print()

    print('=' * 66)
    for label, ok in results:
        print('  %-12s %s' % (label, 'OK' if ok else 'HONG'))
    return 0 if all(ok for _, ok in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
