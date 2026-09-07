#!/usr/bin/env python3
"""Sweep the class-fusion weight with and without the margin gate.

The cumulative ablation removes the gate while holding beta at 0.5, which is
the value that was tuned with the gate in place. That is the standard shape for
a cumulative ablation, but it cannot answer the question it raises: is the gate
worth keeping, or was beta simply too large for an ungated head?

So both arms are swept over the same values of beta, on all six backbones, and
each is judged at its own best setting. Anything else compares a tuned
configuration against an untuned one.

The grid reaches down to beta=0.0 because the ungated arm peaked at the smallest
value of the original grid, 0.3, on five of six backbones. An optimum on the
boundary is not an optimum, and the direction it points matters: beta=0.0 is no
class fusion at all, so without that anchor an ungated win cannot be told apart
from a preference for not fusing. Nothing is gated at beta=0.0, so that column
is measured once, under the ungated arm; the other arms leave it empty rather
than copy it.

Two things about how a winner is picked, both of which this tool got wrong
before and reported backwards:

  * Loss and Forgetting are better when smaller. An earlier version took the
    maximum of every metric, so those two tables named the worst beta as best
    and then counted the head-to-head wins with the sign flipped.

  * Letting each metric choose its own beta does not describe anything that can
    be shipped: the ungated arm minimises Loss at beta=0.2 and maximises Acc@1
    at beta=0.3, and only one of the two can be released. --pick fixes each
    arm's beta by one metric and reads the others at that beta, which is the
    comparison an ablation table actually needs.

Logs are addressed by their exact name, rebuilt from the template
eval_rp_head_any_4090.sh uses; a missing file is reported, never substituted.

Usage:
    python tools/beta_sweep_table.py
    python tools/beta_sweep_table.py --metric Acc@task
    python tools/beta_sweep_table.py --metric Loss --pick Acc@1
"""
import argparse
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET, NUM_TASKS = 'imr', 10

BACKBONES = [
    ('', '', 'Sup-21K'),
    ('mocov3', 'bmocov3', 'MoCo-1K'),
    ('ibot1k', 'bibot', 'iBOT-1K'),
    ('ibot21k', 'b21kibot', 'iBOT-21K'),
    ('dino', 'bdino', 'DINO-1K'),
    ('mae', 'bmae', 'MAE-1K'),
]

BETAS = [('0.0', 'cw0p0'), ('0.1', 'cw0p1'), ('0.2', 'cw0p2'),
         ('0.3', 'cw0p3'), ('0.5', 'cw0p5'), ('0.8', 'cw0p8'),
         ('1.0', 'cw1p0')]
GATES = [('co cong (margin)', 'gmargin'),
         ('co cong (relative)', 'grelative'),
         ('khong cong', '')]

# +1 where a larger number is better, -1 where a smaller one is. Backward is
# normally negative and closer to zero is better, so it counts as larger.
DIRECTION = {'Acc@task': 1, 'Acc@1': 1, 'Acc@5': 1,
             'Loss': -1, 'Forgetting': -1, 'Backward': 1}

FIXED = ('_eval_rp_lora_d10000_relu_l10000_nnone_t0_b0p0_p1_inone_c0_ra0ls0'
         '_f1d1w0p7')


def output_root():
    return os.path.join(os.path.dirname(REPO_ROOT), 'hrm-pet-output')


def value(root, dir_tag, log_tag, class_weight, gate, metric):
    suffix = '_%s' % dir_tag if dir_tag else ''
    base = '%s%s_lora_rank8_baseline_%dtasks_seed42' % (
        DATASET, suffix, NUM_TASKS)
    path = os.path.join(root, '%s%slsw0p0c0ca0%ssh1p0m1%s%s.log'
                        % (base, FIXED, class_weight, gate, log_tag))
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


def choose(pairs, sign):
    """The best (value, beta), ties going to the smaller beta.

    A tie broken toward less fusion keeps the reported setting the more
    conservative of two equals, and keeps Acc@task -- which class fusion cannot
    move at all, so every beta ties -- from naming beta=1.0 as its optimum.
    """
    return min(pairs, key=lambda pair: (-sign * pair[0], float(pair[1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=None)
    parser.add_argument('--metric', default='Acc@1')
    parser.add_argument('--pick', default=None,
                        help='chon beta cua moi nhanh theo chi so nay roi doc '
                             '--metric tai do; mac dinh chon theo --metric')
    args = parser.parse_args()
    root = args.root or output_root()
    pick = args.pick or args.metric
    sign = DIRECTION.get(args.metric, 1)
    pick_sign = DIRECTION.get(pick, 1)

    print('Quet beta co cong va khong cong, Split-ImageNet-R, seed 42, w=0.7')
    print('chi so: %s (%s la tot hon)'
          % (args.metric, 'nho hon' if sign < 0 else 'lon hon'))
    if pick != args.metric:
        print('beta cua moi nhanh chon theo %s, roi doc %s tai chinh beta do'
              % (pick, args.metric))
    print()

    best = {}
    for gate_label, gate in GATES:
        print(gate_label)
        header = '%-11s' % '' + ''.join('%9s' % ('b=' + b) for b, _ in BETAS)
        header += '%11s%9s' % ('tot nhat', 'tai b')
        print(header)
        print('-' * len(header))
        for dir_tag, log_tag, name in BACKBONES:
            cells, shown, chosen = [], {}, []
            for beta, class_weight in BETAS:
                got = value(root, dir_tag, log_tag, class_weight, gate,
                            args.metric)
                shown[beta] = got
                cells.append('%9.2f' % got if got is not None else '%9s' % '--')
                if got is None:
                    continue
                if pick == args.metric:
                    chosen.append((got, beta))
                else:
                    by = value(root, dir_tag, log_tag, class_weight, gate, pick)
                    if by is not None:
                        chosen.append((by, beta))
            if chosen:
                _, at = choose(chosen, pick_sign)
                top = shown[at]
                best[(gate_label, name)] = (top, at)
                cells.append('%11.2f%9s' % (top, at))
            else:
                cells.append('%11s%9s' % ('--', '--'))
            print('%-11s%s' % (name, ''.join(cells)))
        print()

    print('Moi nhanh o beta tot nhat cua chinh no:')
    print('%-11s%12s%12s%10s' % ('', 'relative', 'khong cong', 'chenh'))
    print('-' * 45)
    wins = {'co cong': 0, 'khong cong': 0, 'hoa': 0}
    for _, _, name in BACKBONES:
        gated = best.get(('co cong (relative)', name))
        plain = best.get(('khong cong', name))
        if gated is None or plain is None:
            print('%-11s%12s%12s%10s' % (name, '--', '--', '--'))
            continue
        delta = gated[0] - plain[0]
        # The printed delta stays raw so it can be checked against the columns
        # above; the win goes to whichever direction this metric calls better.
        improvement = sign * delta
        if improvement > 0:
            wins['co cong'] += 1
        elif improvement < 0:
            wins['khong cong'] += 1
        else:
            wins['hoa'] += 1
        print('%-11s%9.2f(%s)%9.2f(%s)%+10.2f'
              % (name, gated[0], gated[1], plain[0], plain[1], delta))
    print('\nco cong thang %d, khong cong thang %d, hoa %d'
          % (wins['co cong'], wins['khong cong'], wins['hoa']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
