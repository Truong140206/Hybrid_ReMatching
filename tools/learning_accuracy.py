#!/usr/bin/env python3
"""Split Backward transfer into the two accuracies it subtracts.

Backward averages a_i(T) - a_i(i): final accuracy on task i minus the accuracy
on task i measured the moment it was learned. Only the first term is retention.
The second is plasticity, and it enters with a minus sign, so a method that
learns each task better is charged for it -- the metric improves when the
subtrahend falls.

On ImageNet-R Sup-21K that is exactly what happens. Against conventional
HRM-PET the hybrid learns each task 1.58 points better on average and keeps
1.38 points more at the end, and the eval script still stamps
`Backward delta=-0.1958: FAIL`, because 1.38 - 1.58 is negative. The gated
variant scores a better Backward than the ungated one for the same reason in
reverse: its learning accuracy is lower on nine of ten tasks.

So Backward cannot be read on its own here. This prints both halves for every
cell, and checks the identity that ties them together:

    delta(Backward) = delta(retention) - delta(plasticity)

a_i(i) is read off the diagonal of the evaluation schedule. Stage t evaluates
tasks 1..t, so among the '* Acc@task ...' lines the diagonal sits at positions
t(t+1)/2, and the final row a_i(T) is the last T of them.

The comparison needs no baseline of a particular kind, only two schedules, so
--vs takes either the conventional HRM-PET log or another arm. That matters
here: the conventional log exists for one backbone on two benchmarks, while the
gated and ungated arms exist on all twenty-one cells, and the claim being
checked is about those two.

Usage:
    python tools/learning_accuracy.py --arm nogate03
    python tools/learning_accuracy.py --arm relative --vs nogate03
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_results as cr

ROW = re.compile(r'^\*\s+Acc@task\s+[0-9.]+\s+Acc@1\s+(-?[0-9.]+)')


def accuracies(path, num_tasks):
    """(diagonal a_t(t), final row a_i(T)), or None if the log is not complete.

    A run that stopped early leaves a short schedule, and reading a diagonal out
    of it would silently mix stages. The expected count is exact, so check it.
    """
    if not path or not os.path.isfile(path):
        return None
    values = []
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            found = ROW.match(line)
            if found:
                values.append(float(found.group(1)))
    if len(values) != num_tasks * (num_tasks + 1) // 2:
        return None
    diagonal = [values[t * (t + 1) // 2 - 1] for t in range(1, num_tasks + 1)]
    return diagonal, values[-num_tasks:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--arm', default='nogate03',
                        help='nhanh can cham: %s' % ', '.join(sorted(cr.ARMS)))
    parser.add_argument('--vs', default='conventional',
                        help="cham voi cai gi: 'conventional' la HRM-PET goc, "
                             'hoac ten mot nhanh khac')
    args = parser.parse_args()
    root = args.root or cr.output_root()
    for arm in (args.arm, args.vs):
        if arm not in cr.ARMS and arm != 'conventional':
            print("khong biet nhanh %r; co: conventional, %s"
                  % (arm, ', '.join(sorted(cr.ARMS))))
            return 1

    against = 'HRM-PET goc' if args.vs == 'conventional' else args.vs
    print('Tach Backward thanh hai nua, %s so voi %s' % (args.arm, against))
    print('dau duong nghia la %s hon %s o cot do\n' % (args.arm, against))
    print('%-12s %-11s %8s %8s %9s %9s'
          % ('bo du lieu', 'backbone', 'hoc', 'giu', 'giu-hoc', 'backward'))
    print('%-12s %-11s %8s %8s %9s %9s'
          % ('', '', 'a_i(i)', 'a_i(T)', 'suy ra', 'do duoc'))
    print('-' * 62)

    rows = []
    for dataset, dataset_label, num_tasks in cr.DATASETS:
        for dir_tags, log_tag, backbone_label in cr.BACKBONES:
            arm_path = cr.find_log(root, dataset, dir_tags, log_tag, num_tasks,
                                   args.arm)
            if args.vs == 'conventional':
                base_path = None
                for dir_tag in dir_tags:
                    suffix = '_%s' % dir_tag if dir_tag else ''
                    candidate = os.path.join(
                        root, '%s%s_lora_rank8_baseline_%dtasks_seed42'
                              '_eval_conventional.log'
                              % (dataset, suffix, num_tasks))
                    if os.path.isfile(candidate):
                        base_path = candidate
                        break
            else:
                base_path = cr.find_log(root, dataset, dir_tags, log_tag,
                                        num_tasks, args.vs)
            ours = accuracies(arm_path, num_tasks)
            base = accuracies(base_path, num_tasks)
            if ours is None or base is None:
                continue
            # The identity is only worth printing if it is checked against the
            # Backward the run actually reported, not against itself.
            ours_row = cr.final_row(arm_path, num_tasks)
            base_row = cr.final_row(base_path, num_tasks)
            if not ours_row or not base_row:
                continue
            measured = ours_row.get('Backward')
            baseline = base_row.get('Backward')
            if measured is None or baseline is None:
                continue
            # Backward averages over the tasks that have a later stage, so the
            # last task is excluded from both halves.
            keep = num_tasks - 1
            learn = (sum(ours[0][:keep]) - sum(base[0][:keep])) / keep
            retain = (sum(ours[1][:keep]) - sum(base[1][:keep])) / keep
            print('%-12s %-11s %+8.2f %+8.2f %+9.2f %+9.2f'
                  % (dataset_label, backbone_label, learn, retain,
                     retain - learn, measured - baseline))
            rows.append((learn, retain, measured - baseline))
        print()

    if not rows:
        print('khong o nao co du ca hai log')
        return 1
    learn = sum(r[0] for r in rows) / len(rows)
    retain = sum(r[1] for r in rows) / len(rows)
    drift = max(abs(r[1] - r[0] - r[2]) for r in rows)
    print('%d o. Trung binh: hoc %+.2f, giu %+.2f, nen Backward %+.2f.'
          % (len(rows), learn, retain, retain - learn))
    print('Sai lech lon nhat giua hai cot cuoi: %.3f diem.' % drift)
    # Counting both directions keeps the summary readable whichever arm is the
    # stronger one; an earlier version assumed --arm was the better of the two
    # and its sentence came out backwards when it was not.
    print('hoc kem hon o %d/%d o, giu kem hon o %d/%d o.'
          % (sum(1 for a, _, _ in rows if a < 0), len(rows),
             sum(1 for _, b, _ in rows if b < 0), len(rows)))
    print('Chenh Backward %+.2f tach ra: %+.2f tu giu, %+.2f tu hoc.'
          % (retain - learn, retain, -learn))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
