#!/usr/bin/env python3
"""Measure task-recency bias in the routed head, and what removing it is worth.

Every fusion rule tried so far applies the same function at every stage, so it
moves a_i(i) and a_i(T) together and cancels out of Forgetting and Backward,
which are differences of the two. Improving those metrics honestly needs an
effect whose harm grows with the number of tasks seen -- larger at stage T than
at stage i.

Task-recency bias has exactly that shape. If the shared classifier assigns
systematically higher scores to the classes of recent tasks, then at stage i the
task being measured is the newest and barely suffers, while at stage T it is the
oldest and suffers most. Removing it should therefore raise a_i(T) much more than
a_i(i), which is what Forgetting rewards.

Whether the bias exists here is a measurement, not an assumption, so this reports
the per-block score profile before it corrects anything. The class-to-task map is
recovered exactly from the dump: every sample carries both its label and its task.

The correction is per sample and uses no labels and no test-set statistics: each
task block's scores are centred on that sample's own block means, so a block
cannot win by sitting higher than the others. alpha=0 leaves the scores untouched
and reproduces the run exactly.

Usage:
    python tools/task_bias.py dump.npz
    python tools/task_bias.py dump.npz --beta 0.3
"""
import argparse

import numpy as np


def standardize(scores, valid):
    out = np.where(valid, scores, np.nan)
    mean = np.nanmean(out, axis=1, keepdims=True)
    std = np.nanstd(out, axis=1, keepdims=True)
    return np.where(valid, (out - mean) / np.maximum(std, 1e-6), -1e4)


def accuracy_by_task(scores, target, task):
    correct = scores.argmax(axis=1) == target
    return float(np.mean([correct[task == t].mean()
                          for t in np.unique(task)]) * 100.0)


def per_task_accuracy(scores, target, task):
    correct = scores.argmax(axis=1) == target
    return np.array([correct[task == t].mean() * 100.0
                     for t in np.unique(task)])


def class_to_task(target, task, width):
    """Which task owns each class, read off the labels themselves."""
    owner = np.full(width, -1, dtype=int)
    for value, block in zip(target, task):
        owner[value] = block
    return owner


def debias(scores, owner, blocks, alpha):
    """Centre each task block on its own per-sample mean, by alpha."""
    out = scores.copy()
    for block in blocks:
        columns = owner == block
        if not columns.any():
            continue
        means = out[:, columns].mean(axis=1, keepdims=True)
        out[:, columns] -= alpha * means
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump')
    parser.add_argument('--beta', type=float, default=0.3)
    args = parser.parse_args()

    data = np.load(args.dump)
    valid = data['valid']
    target = data['target'].astype(int)
    task = data['task'].astype(int)
    routed = standardize(data['routed'].astype(np.float64), valid)
    rp = standardize(data['rp'].astype(np.float64), valid)
    owner = class_to_task(target, task, routed.shape[1])
    blocks = np.unique(task)

    print('%d mau, %d nhiem vu, %d lop co chu\n'
          % (len(target), len(blocks), int((owner >= 0).sum())))

    # The profile first: if recent blocks do not score higher, there is no bias
    # to remove and the rest of this tool is answering a question nobody asked.
    print('Diem trung binh moi khoi nhiem vu, tren moi mau (da chuan hoa):')
    print('%-6s %10s %10s %10s' % ('nv', 'routed', 'rp', 'do chinh xac'))
    print('-' * 40)
    accuracy = per_task_accuracy(
        (1.0 - args.beta) * routed + args.beta * rp, target, task)
    for index, block in enumerate(blocks):
        columns = owner == block
        print('%-6d %10.4f %10.4f %10.2f'
              % (block, routed[:, columns].mean(), rp[:, columns].mean(),
                 accuracy[index]))

    early, late = blocks[:len(blocks) // 2], blocks[len(blocks) // 2:]
    gap = (routed[:, np.isin(owner, late)].mean()
           - routed[:, np.isin(owner, early)].mean())
    print('\nKhoi moi cao hon khoi cu: %+.4f (routed), %+.4f (rp)'
          % (gap, rp[:, np.isin(owner, late)].mean()
             - rp[:, np.isin(owner, early)].mean()))

    print('\nSua thien lech, quet cuong do alpha:')
    print('%-7s %9s %11s %11s' % ('alpha', 'Acc@1', 'nv cu', 'nv moi'))
    print('-' * 41)
    base = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        fixed = debias(routed, owner, blocks, alpha)
        mixed = (1.0 - args.beta) * fixed + args.beta * rp
        each = per_task_accuracy(mixed, target, task)
        row = (accuracy_by_task(mixed, target, task),
               each[:len(blocks) // 2].mean(), each[len(blocks) // 2:].mean())
        if base is None:
            base = row
        print('%-7.2f %9.2f %11.2f %11.2f' % (alpha,) + '' % ())
        print('\033[F%-7.2f %9.2f %11.2f %11.2f'
              % (alpha, row[0], row[1], row[2]))
    print('\nChenh so voi alpha=0 la thu dang doc: neu loi ich don vao cot '
          '"nv cu"\nthi hieu ung co dung hinh dang de nang a_i(T) hon a_i(i).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
