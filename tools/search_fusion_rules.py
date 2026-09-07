#!/usr/bin/env python3
"""Search offline for a rule that chooses between the routed head and the RP head.

The class-union audit bounds what any per-sample choice between the two can add:
+1.97 Acc@1 on ImageNet-R Sup-21K, +5.22 on MAE. Three hand-designed gates --
margin, margin_both, relative -- captured none of it, and each cost a four-minute
evaluation to find that out. Five null results in a day is what testing one
hypothesis per GPU run buys.

So --rp_dump_scores writes the final stage's per-sample scores once, and this
searches the space against them in seconds.

The question it answers first is not which rule wins but whether any signal
exists. On the samples where exactly one head is right, a feature that cannot
separate "RP is the right one" from "routed is the right one" cannot be the
basis of any gate, however it is thresholded. That is an AUC, and an AUC near
0.5 across every candidate is a stopping condition, not a prompt to try a
fourth hand-designed function.

Anything fitted is fitted on half the samples and scored on the other half.
Reading a rule's quality off the samples that chose it is how an oracle gets
mistaken for a method.

Usage:
    python tools/search_fusion_rules.py dump.npz
    python tools/search_fusion_rules.py dump.npz --beta 0.3
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


def accuracy(scores, target):
    return float((scores.argmax(axis=1) == target).mean() * 100.0)


def auc(scores, labels):
    """Rank AUC, ties handled by average rank; 0.5 means no separation."""
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True,
                                        return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
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
    design = np.hstack([(features - mean) / scale,
                        np.ones((len(features), 1))])
    weights = np.zeros(design.shape[1])
    for _ in range(steps):
        prediction = 1.0 / (1.0 + np.exp(-design @ weights))
        weights -= rate * design.T @ (prediction - labels) / len(labels)
    return mean, scale, weights


def apply_logistic(features, model):
    mean, scale, weights = model
    design = np.hstack([(features - mean) / scale,
                        np.ones((len(features), 1))])
    return 1.0 / (1.0 + np.exp(-design @ weights))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dump')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='trong so dang dung, de doi chieu voi log')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.dump)
    valid = data['valid']
    target = data['target'].astype(int)
    routed = standardize(data['routed'].astype(np.float64), valid)
    rp = standardize(data['rp'].astype(np.float64), valid)
    count = len(target)

    routed_ok = routed.argmax(axis=1) == target
    rp_ok = rp.argmax(axis=1) == target
    fused = (1.0 - args.beta) * routed + args.beta * rp

    print('%d mau, %d lop hop le trung binh\n'
          % (count, valid.sum(axis=1).mean()))
    print('%-34s %8s' % ('cau hinh', 'Acc@1'))
    print('-' * 43)
    print('%-34s %8.2f' % ('chi dinh tuyen (beta=0)', accuracy(routed, target)))
    print('%-34s %8.2f' % ('chi RP (beta=1)', accuracy(rp, target)))
    print('%-34s %8.2f' % ('tron hien tai (beta=%.2f)' % args.beta,
                           accuracy(fused, target)))
    print('%-34s %8.2f' % ('tran: hop oracle',
                           float((routed_ok | rp_ok).mean() * 100.0)))
    print('%-34s %8.2f' % ('  trong do RP dung mot minh',
                           float((rp_ok & ~routed_ok).mean() * 100.0)))
    print('%-34s %8.2f' % ('  trong do dinh tuyen dung mot minh',
                           float((routed_ok & ~rp_ok).mean() * 100.0)))

    print('\nQuet beta hang so:')
    best_beta, best_acc = None, -1.0
    for beta in np.arange(0.0, 1.01, 0.05):
        got = accuracy((1.0 - beta) * routed + beta * rp, target)
        if got > best_acc:
            best_beta, best_acc = beta, got
    print('  tot nhat %.2f tai beta=%.2f' % (best_acc, best_beta))

    # Features are built from what a gate could actually see at inference.
    routed_top1, routed_top2 = top2(routed)
    rp_top1, rp_top2 = top2(rp)
    margin_routed = routed_top1 - routed_top2
    margin_rp = rp_top1 - rp_top2
    features = {
        'bien routed': margin_routed,
        'bien rp': margin_rp,
        'hieu bien (rp - routed)': margin_rp - margin_routed,
        'ti le bien': margin_rp / np.maximum(margin_rp + margin_routed, 1e-6),
        'xac suat dinh routed': softmax_top(routed),
        'xac suat dinh rp': softmax_top(rp),
        'entropy routed': normalised_entropy(routed, valid),
        'entropy rp': normalised_entropy(rp, valid),
        'hang cua dinh rp trong routed': np.log1p(
            rank_of(rp.argmax(axis=1), routed)),
        'hang cua dinh routed trong rp': np.log1p(
            rank_of(routed.argmax(axis=1), rp)),
        'hai dau dong y': (routed.argmax(axis=1)
                           == rp.argmax(axis=1)).astype(float),
    }

    # Only the samples where exactly one head is right carry any information
    # about which to believe; everywhere else the choice cannot change the
    # answer, so scoring a rule on them would flatter it.
    decisive = rp_ok ^ routed_ok
    labels = rp_ok[decisive].astype(float)
    print('\n%d mau ma dung mot dau dung (%.2f%% cua tap), trong do RP dung '
          '%.1f%%' % (decisive.sum(), 100.0 * decisive.mean(),
                      100.0 * labels.mean()))
    print('\nAUC tach "RP dung" khoi "dinh tuyen dung", tren tap do:')
    print('%-34s %8s' % ('dac trung', 'AUC'))
    print('-' * 43)
    ranked = sorted(((auc(v[decisive], labels), k)
                     for k, v in features.items()), reverse=True)
    for value, name in ranked:
        print('%-34s %8.3f' % (name, value))

    rng = np.random.default_rng(args.seed)
    split = rng.random(count) < 0.5
    matrix = np.column_stack([features[k] for k in features])
    train = decisive & split
    test = ~split
    model = fit_logistic(matrix[train], rp_ok[train].astype(float))
    print('\nKet hop tuyen tinh moi dac trung, khop tren nua nay, cham tren nua kia:')
    print('  AUC tren nua kiem tra: %.3f'
          % auc(apply_logistic(matrix[decisive & test], model),
                rp_ok[decisive & test].astype(float)))

    probability = apply_logistic(matrix, model)
    print('\n%-34s %8s %8s' % ('luat', 'Acc@1', 'so voi tron'))
    print('-' * 52)
    reference = accuracy(fused[test], target[test])
    print('%-34s %8.2f %8s' % ('tron hien tai (nua kiem tra)', reference, '--'))
    for scale in (0.3, 0.5, 0.8, 1.0):
        weights = (scale * probability)[:, None]
        mixed = (1.0 - weights) * routed + weights * rp
        got = accuracy(mixed[test], target[test])
        print('%-34s %8.2f %+8.2f'
              % ('beta_i = %.1f * p(RP dung)' % scale, got, got - reference))
    for threshold in (0.5, 0.6, 0.7):
        pick = probability > threshold
        mixed = np.where(pick[:, None], rp, fused)
        got = accuracy(mixed[test], target[test])
        print('%-34s %8.2f %+8.2f'
              % ('chon RP khi p > %.1f' % threshold, got, got - reference))
    oracle = np.where((rp_ok & ~routed_ok)[:, None], rp, fused)
    print('%-34s %8.2f %+8.2f'
          % ('oracle (biet truoc)', accuracy(oracle[test], target[test]),
             accuracy(oracle[test], target[test]) - reference))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
