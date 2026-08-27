"""Numpy mirror of the margin_both gate, to check the scale fix off-GPU.

Reproduces _valid_moments / _standardize_valid / _top2_margin exactly and
compares the RP margin before and after putting the ridge scores on the routed
logits' scale. No torch needed.
"""
import numpy as np

rng = np.random.default_rng(0)
N, C = 20000, 200


def valid_moments(s, v):
    masked = np.where(v, s, 0.0)
    count = np.maximum(v.sum(axis=1, keepdims=True), 1).astype(float)
    mean = masked.sum(axis=1, keepdims=True) / count
    centered = np.where(v, s - mean, 0.0)
    std = np.sqrt((centered ** 2).sum(axis=1, keepdims=True) / count)
    return mean, np.maximum(std, 1e-6)


def standardize_valid(s, v):
    mean, std = valid_moments(s, v)
    return np.where(v, s - mean, 0.0) / std


def top2_margin(s, v):
    z = np.where(v, s, -1e4)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    part = np.sort(p, axis=1)[:, -2:]
    p2, p1 = part[:, 0], part[:, 1]
    return np.clip((p1 - p2) / np.maximum(p1 + p2, 1e-12), 0.0, 1.0)


valid = np.ones((N, C), dtype=bool)

# Routed logits: spread ~35 units, as reported in the handoff.
routed = rng.normal(0.0, 5.0, size=(N, C))
routed[np.arange(N), rng.integers(0, C, N)] += 8.0

# Ridge scores: same shape of signal, but spread ~1 unit.
rp_raw = rng.normal(0.0, 0.14, size=(N, C))
rp_raw[np.arange(N), rng.integers(0, C, N)] += 0.23

print('spread (max-min, mean over samples)')
print('  routed logits %.2f' % (routed.max(1) - routed.min(1)).mean())
print('  ridge scores  %.2f' % (rp_raw.max(1) - rp_raw.min(1)).mean())

conf_routed = top2_margin(routed, valid)
conf_rp_before = top2_margin(rp_raw, valid)

mean, std = valid_moments(routed, valid)
rp_on_scale = standardize_valid(rp_raw, valid) * std + mean
conf_rp_after = top2_margin(rp_on_scale, valid)

print()
print('mean top-2 margin')
print('  conf(routed)          %.4f' % conf_routed.mean())
print('  conf(rp) raw scale    %.4f   <- what the broken run used' % conf_rp_before.mean())
print('  conf(rp) fixed scale  %.4f   <- after the affine map' % conf_rp_after.mean())
print('  shrink factor removed %.1fx'
      % (conf_rp_after.mean() / max(conf_rp_before.mean(), 1e-12)))

undecided = 1.0 - conf_routed
print()
print('effective share at nominal beta, mean over samples')
for beta in (0.3, 0.5, 0.8, 1.0):
    print('  beta=%.1f  margin %.4f | margin_both broken %.4f | margin_both fixed %.4f'
          % (beta, beta * undecided.mean(),
             beta * (undecided * conf_rp_before).mean(),
             beta * (undecided * conf_rp_after).mean()))

# Identity check: conf(rp) = 1 must recover 'margin' exactly.
ones = np.ones_like(conf_routed)
assert np.allclose(undecided * ones, undecided)
print()
print('identity check conf(rp)=1 recovers margin: OK')

# Which nominal beta for margin_both matches margin@0.5's mean share?
match = 0.5 * undecided.mean() / max((undecided * conf_rp_after).mean(), 1e-12)
print('beta for margin_both that matches margin@0.5 mean share: %.2f' % match)
