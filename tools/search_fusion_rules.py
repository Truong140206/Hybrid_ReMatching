#!/usr/bin/env python3
"""Search offline for a rule that chooses between the routed head and the RP head.

The class-union audit bounds what any per-sample choice between the two can add:
+2.15 Acc@1 on ImageNet-R Sup-21K, +5.33 on MAE. Three hand-designed gates --
margin, margin_both, relative -- captured none of it, and each cost a four-minute
evaluation to find that out. Five null results in a day is what one hypothesis
per GPU run buys, so --rp_dump_scores writes the final stage's scores once and
the search happens here in seconds.

What that turned up contradicts the reason the gates were abandoned. There IS a
signal: the difference of the two top-2 margins separates "the RP head is the
right one" from "the routed head is the right one" at AUC 0.72 and 0.75, and a
linear combination of every feature reaches 0.78 and 0.83 on held-out samples.
The gates did not fail because the two heads are indistinguishable.

They failed because a signal is not a gain. On Sup-21K the routed head is right
alone on 8.63% of samples and the RP head on 4.48%: trusting the routed head is
already close to optimal, and even a good discriminator adds +0.13. On MAE the
split is 10.88 against 12.30 -- balanced -- and the same discriminator adds
+1.23. A gate has room only where the two heads are comparably strong, and every
gate here was designed and judged on Sup-21K, where none can help.

Two things this must not be read as saying. The fitted selector is fitted on half
the test labels, so it bounds a learnable gate rather than being one. And a rule
that does not transfer is not a rule: --holdout-tasks fits on the early tasks and
scores on the late ones, --apply-to fits on one dump and scores on another, and a
rule surviving neither is a description of these samples.

Usage:
    python tools/search_fusion_rules.py dump.npz
    python tools/search_fusion_rules.py dump.npz --holdout-tasks
    python tools/search_fusion_rules.py dump_a.npz --apply-to dump_b.npz
"""
import argparse

import numpy as np


def standardize(scores, valid):
    """Per sample, over the valid classes only -- what the engine mixes."""
    out = np.where(valid, scores, np.nan)
    mean = np.nanmean(out, axis=1, keepdims=True)
    std = np.nanstd(out, axis=1, keepdims=True)
    return np.where(valid, (out - mean) / np.maximum(std, 1e-6), -1e4)


def top2(scores):
    part = np.partition(scores, -2, axis=1)
    return part[:, -1], part[:, -2]


def softmax_top(scores):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)


def normalised_entropy(scores, valid):
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted) * valid
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    ent = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    return ent / np.log(np.maximum(valid.sum(axis=1), 2))


def rank_of(target_index, scores):
    """Where each sample's target_index sits in that row's ordering, 0 = top."""
    chosen = scores[np.arange(scores.shape[0]), target_index]
    return (scores > chosen[:, None]).sum(axis=1)


def accuracy_by_task(scores, target, task, mask=None):
    """Mean of the per-task accuracies, which is what the logs report.

    The final row averages ten per-task numbers rather than pooling six thousand
    samples, and ImageNet-R's tasks hold different numbers of test images, so the
    two differ -- by 0.11 on Sup-21K and 0.52 on MAE. Pooling and calling it the
    same number is how a tool silently disagrees with the run it reproduces.
    """
    correct = scores.argmax(axis=1) == target
    if mask is not None:
        correct, task = correct[mask], task[mask]
    return float(np.mean([correct[task == t].mean()
                          for t in np.unique(task)]) * 100.0)


def auc(scores, labels):
    """Rank AUC, ties at average rank; 0.5 means no separation at all."""
    order = np.argsort(scores, kind='mergesort')
    ordinal = np.empty(len(scores), dtype=float)
    ordinal[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True,
                                        return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ordinal)
    ranks = (sums / counts)[inverse]
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    return float((ranks[labels == 1].sum()
                  - positives * (positives + 1) / 2) / (positives * negatives))


def fit_logistic(features, labels, steps=400, rate=0.5):
    """Plain gradient descent, so the tool needs nothing beyond numpy."""
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    design = np.hstack([(features - mean) / scale, np.ones((len(features), 1))])
    weights = np.zeros(design.shape[1])
    for _ in range(steps):
        prediction = 1.0 / (1.0 + np.exp(-design @ weights))
        weights -= rate * design.T @ (prediction - labels) / len(labels)
    return mean, scale, weights


def apply_logistic(features, model):
    mean, scale, weights = model
    design = np.hstack([(features - mean) / scale, np.ones((len(features), 1))])
    return 1.0 / (1.0 + np.exp(-design @ weights))


FEATURE_NAMES = (
    'bien routed', 'bien rp', 'hieu bien (rp - routed)', 'ti le bien',
    'xac suat dinh routed', 'xac suat dinh rp', 'entropy routed', 'entropy rp',
    'hang cua dinh rp trong routed', 'hang cua dinh routed trong rp',
    'hai dau dong y')


def prepare(path, beta):
    """Everything the search needs, from one dump."""
    data = np.load(path)
    valid = data['valid']
    target = data['target'].astype(int)
    task = data['task'].astype(int)
    routed = standardize(data['routed'].astype(np.float64), valid)
    rp = standardize(data['rp'].astype(np.float64), valid)

    routed_top1, routed_top2 = top2(routed)
    rp_top1, rp_top2 = top2(rp)
    margin_routed = routed_top1 - routed_top2
    margin_rp = rp_top1 - rp_top2
    columns = (
        margin_routed,
        margin_rp,
        margin_rp - margin_routed,
        margin_rp / np.maximum(margin_rp + margin_routed, 1e-6),
        softmax_top(routed),
        softmax_top(rp),
        normalised_entropy(routed, valid),
        normalised_entropy(rp, valid),
        np.log1p(rank_of(rp.argmax(axis=1), routed)),
        np.log1p(rank_of(routed.argmax(axis=1), rp)),
        (routed.argmax(axis=1) == rp.argmax(axis=1)).astype(float),
    )
    return {
        'valid': valid, 'target': target, 'task': task,
        'routed': routed, 'rp': rp,
        'fused': (1.0 - beta) * routed + beta * rp,
        'routed_ok': routed.argmax(axis=1) == target,
        'rp_ok': rp.argmax(axis=1) == target,
        'features': np.column_stack(columns),
        'name': path.rsplit('/', 1)[-1],
    }


def score_rules(bag, model, mask, title):
    """What the fitted probability is worth, scored only on `mask`."""
    target, task, fused = bag['target'], bag['task'], bag['fused']
    probability = apply_logistic(bag['features'], model)
    reference = accuracy_by_task(fused, target, task, mask)
    print('\n%s' % title)
    print('%-34s %8s %11s' % ('luat', 'Acc@1', 'so voi tron'))
    print('-' * 55)
    print('%-34s %8.2f %11s' % ('tron hien tai', reference, '--'))
    for scale in (0.5, 0.8, 1.0):
        weights = (scale * probability)[:, None]
        mixed = (1.0 - weights) * bag['routed'] + weights * bag['rp']
        got = accuracy_by_task(mixed, target, task, mask)
        print('%-34s %8.2f %+11.2f'
              % ('beta_i = %.1f * p(RP dung)' % scale, got, got - reference))
    for threshold in (0.5, 0.6):
        mixed = np.where((probability > threshold)[:, None], bag['rp'], fused)
        got = accuracy_by_task(mixed, target, task, mask)
        print('%-34s %8.2f %+11.2f'
              % ('chon RP khi p > %.1f' % threshold, got, got - reference))
    oracle = np.where((bag['rp_ok'] & ~bag['routed_ok'])[:, None],
                      bag['rp'], fused)
    ceiling = accuracy_by_task(oracle, target, task, mask)
    print('%-34s %8.2f %+11.2f'
          % ('oracle (biet truoc)', ceiling, ceiling - reference))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='trong so dang dung, de doi chieu voi log')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--holdout-tasks', action='store_true',
                        help='khop tren nua nhiem vu dau, cham tren nua sau')
    parser.add_argument('--apply-to', default=None,
                        help='khop tren dump nay roi cham tren dump kia')
    args = parser.parse_args()

    bag = prepare(args.dump, args.beta)
    target, task = bag['target'], bag['task']
    routed_ok, rp_ok = bag['routed_ok'], bag['rp_ok']
    count = len(target)

    print('%d mau, %d lop hop le trung binh\n'
          % (count, bag['valid'].sum(axis=1).mean()))
    print('%-34s %8s %9s' % ('cau hinh', 'gop', 'theo nv'))
    print('%-34s %8s %9s' % ('', '%d mau' % count, 'nhu log'))
    print('-' * 53)
    for name, scores in (('chi dinh tuyen (beta=0)', bag['routed']),
                         ('chi RP (beta=1)', bag['rp']),
                         ('tron hien tai (beta=%.2f)' % args.beta,
                          bag['fused'])):
        pooled = float((scores.argmax(axis=1) == target).mean() * 100.0)
        print('%-34s %8.2f %9.2f'
              % (name, pooled, accuracy_by_task(scores, target, task)))
    print('%-34s %8.2f' % ('tran: hop oracle',
                           float((routed_ok | rp_ok).mean() * 100.0)))
    print('%-34s %8.2f' % ('  RP dung mot minh',
                           float((rp_ok & ~routed_ok).mean() * 100.0)))
    print('%-34s %8.2f' % ('  dinh tuyen dung mot minh',
                           float((routed_ok & ~rp_ok).mean() * 100.0)))

    # Only the samples where exactly one head is right carry information about
    # which to believe; elsewhere the choice cannot change the answer, so
    # scoring a rule on them would flatter it.
    decisive = rp_ok ^ routed_ok
    labels = rp_ok[decisive].astype(float)
    print('\n%d mau ma dung mot dau dung (%.2f%% cua tap), RP dung %.1f%% '
          'trong so do' % (decisive.sum(), 100.0 * decisive.mean(),
                           100.0 * labels.mean()))
    print('\nAUC tach "RP dung" khoi "dinh tuyen dung":')
    print('%-34s %8s' % ('dac trung', 'AUC'))
    print('-' * 43)
    for value, name in sorted(
            ((auc(bag['features'][decisive, k], labels), FEATURE_NAMES[k])
             for k in range(len(FEATURE_NAMES))), reverse=True):
        print('%-34s %8.3f' % (name, value))

    rng = np.random.default_rng(args.seed)
    split = rng.random(count) < 0.5
    model = fit_logistic(bag['features'][decisive & split],
                         rp_ok[decisive & split].astype(float))
    print('\nAUC cua ket hop tuyen tinh tren nua chua thay: %.3f'
          % auc(apply_logistic(bag['features'][decisive & ~split], model),
                rp_ok[decisive & ~split].astype(float)))
    score_rules(bag, model, ~split,
                'Chia ngau nhien: khop nua nay, cham nua kia')

    if args.holdout_tasks:
        # Fitting on tasks already seen is the only fit available at inference
        # time, so this transfer decides whether the rule is a mechanism or a
        # description of these particular samples.
        middle = np.median(np.unique(task))
        early, late = task <= middle, task > middle
        model = fit_logistic(bag['features'][decisive & early],
                             rp_ok[decisive & early].astype(float))
        print('\nAUC khop tren nhiem vu <= %g, cham tren nhiem vu > %g: %.3f'
              % (middle, middle,
                 auc(apply_logistic(bag['features'][decisive & late], model),
                     rp_ok[decisive & late].astype(float))))
        score_rules(bag, model, late,
                    'Chuyen qua nhiem vu: khop nhiem vu dau, cham nhiem vu sau')

    if args.apply_to:
        other = prepare(args.apply_to, args.beta)
        model = fit_logistic(bag['features'][decisive],
                             rp_ok[decisive].astype(float))
        other_decisive = other['rp_ok'] ^ other['routed_ok']
        print('\nAUC khop tren %s, cham tren %s: %.3f'
              % (bag['name'], other['name'],
                 auc(apply_logistic(other['features'][other_decisive], model),
                     other['rp_ok'][other_decisive].astype(float))))
        score_rules(other, model, np.ones(len(other['target']), bool),
                    'Chuyen qua backbone: khop dump nay, cham dump kia')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
