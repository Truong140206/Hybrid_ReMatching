"""Property tests for the RP head and the two fusion stages.

These cover the contribution itself, which until now had no tests at all: the
21 files beside this one all belong to directions that were abandoned.

Every assertion here restates a property the source already claims in prose.
The one that matters most is test_margin_both_gate_is_scale_invariant. The first
margin_both implementation ran a softmax straight on the raw ridge scores, which
span about one unit while the routed logits span about thirty-five, so every RP
margin came out near 0.05 and the gate was roughly twenty times too small. That
mode then under-fused in silence, and it cost a full four-point beta sweep on the
GPU to find out. A rescaling of the RP scores must not move the gate, and that
is a five-line check.
"""
from types import SimpleNamespace

import pytest
import torch

from engines.random_projection_head import (
    accumulate_rp_statistics,
    get_rp_state,
    project_features,
    reset_rp_head,
    rp_head_predict,
    rp_head_ready,
    solve_rp_head,
)
from engines.hrm_lora_wtp_and_tap_engine import (
    _fusion_gate,
    _stage_ramp,
    _standardize_valid,
    _top2_margin,
    _valid_moments,
    args_ref,
    fuse_class_scores,
    fuse_routers,
    gate_stats,
)

DEVICE = torch.device('cpu')


def _rp_args(**over):
    args = SimpleNamespace(
        rp_dim=64, rp_seed=1993, rp_activation='relu',
        rp_normalize='none', rp_lambda=1e4)
    for key, value in over.items():
        setattr(args, key, value)
    return args


def _two_heads(batch=256, classes=40, seed=0):
    """Routed logits and ridge scores on the scales the real system produces."""
    generator = torch.Generator().manual_seed(seed)
    routed = torch.randn(batch, classes, generator=generator) * 5.0
    routed[torch.arange(batch),
           torch.randint(0, classes, (batch,), generator=generator)] += 8.0
    # Ridge scores span about one unit against the logits' thirty-five.
    rp = torch.randn(batch, classes, generator=generator) * 0.14
    rp[torch.arange(batch),
       torch.randint(0, classes, (batch,), generator=generator)] += 0.23
    return routed, rp


# --------------------------------------------------------------------------
# Gate and class fusion
# --------------------------------------------------------------------------

def test_margin_both_gate_is_scale_invariant():
    """A positive affine rescaling of the RP scores must not move the gate.

    This is the regression test for the scale bug. The broken implementation
    softmaxed the raw ridge scores, so multiplying them by 100 changed the gate
    by more than an order of magnitude.
    """
    routed, rp = _two_heads()
    valid = torch.ones_like(routed, dtype=torch.bool)
    base = _fusion_gate(routed, valid, 'margin_both', rp_scores=rp)
    scaled = _fusion_gate(routed, valid, 'margin_both',
                          rp_scores=rp * 100.0 + 7.0)
    assert torch.allclose(base, scaled, atol=1e-5)


def test_margin_both_gate_is_on_the_same_scale_as_the_routed_margin():
    """conf(rp) must be comparable to conf(routed), not ~20x smaller."""
    routed, rp = _two_heads()
    valid = torch.ones_like(routed, dtype=torch.bool)
    _fusion_gate(routed, valid, 'margin_both', rp_scores=rp)
    conf_routed = _top2_margin(routed, valid).mean().item()
    assert gate_stats['conf_rp'] > 0.3 * conf_routed
    assert gate_stats['conf_rp'] < 3.0 * conf_routed


def test_confident_rp_head_recovers_the_one_sided_margin_gate():
    """Setting conf(rp) = 1 must reproduce 'margin' exactly."""
    routed, _ = _two_heads()
    valid = torch.ones_like(routed, dtype=torch.bool)
    # One class far above the rest on every sample: top-2 margin saturates at 1.
    decided = torch.zeros_like(routed)
    decided[torch.arange(routed.shape[0]), 0] = 1e3
    one_sided = _fusion_gate(routed, valid, 'margin')
    two_sided = _fusion_gate(routed, valid, 'margin_both', rp_scores=decided)
    assert torch.allclose(one_sided, two_sided, atol=1e-6)


def test_two_sided_gate_never_exceeds_the_one_sided_gate():
    routed, rp = _two_heads()
    valid = torch.ones_like(routed, dtype=torch.bool)
    one_sided = _fusion_gate(routed, valid, 'margin')
    two_sided = _fusion_gate(routed, valid, 'margin_both', rp_scores=rp)
    assert torch.all(two_sided <= one_sided + 1e-6)


def test_margin_both_without_rp_scores_raises():
    routed, _ = _two_heads()
    valid = torch.ones_like(routed, dtype=torch.bool)
    with pytest.raises(ValueError, match='margin_both'):
        _fusion_gate(routed, valid, 'margin_both', rp_scores=None)


def test_unknown_gate_mode_raises():
    routed, _ = _two_heads()
    valid = torch.ones_like(routed, dtype=torch.bool)
    with pytest.raises(ValueError, match='khong-co-che-do|rp_class_fusion_gate'):
        _fusion_gate(routed, valid, 'khong-co-che-do')


def test_gate_stats_are_cleared_between_modes():
    routed, rp = _two_heads()
    valid = torch.ones_like(routed, dtype=torch.bool)
    _fusion_gate(routed, valid, 'margin_both', rp_scores=rp)
    assert 'conf_rp' in gate_stats
    _fusion_gate(routed, valid, 'margin')
    assert 'conf_rp' not in gate_stats, 'stale conf_rp leaked into margin mode'


@pytest.mark.parametrize('mode', ['none', 'margin', 'margin_both', 'entropy'])
def test_gate_stays_inside_the_unit_interval(mode):
    routed, rp = _two_heads()
    valid = torch.ones_like(routed, dtype=torch.bool)
    gate = _fusion_gate(routed, valid, mode, rp_scores=rp)
    if mode == 'none':
        assert gate is None
        return
    assert torch.isfinite(gate).all()
    assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0


def test_zero_weight_class_fusion_returns_the_routed_logits_untouched():
    """beta = 0 must be an exact identity, the claim the mixing docstring makes."""
    routed, rp = _two_heads()
    args_ref[0] = SimpleNamespace(
        rp_class_fusion_gate='margin', rp_class_fusion_sharpen=1.0,
        rp_fusion_ramp=0.0, num_tasks=10)
    mixed = fuse_class_scores(routed, rp, 0.0, seen_tasks=10)
    assert torch.allclose(mixed, routed, atol=1e-4)


def test_class_fusion_recentres_on_the_routed_mean_and_shrinks_the_spread():
    """The affine map back restores the mean exactly; the spread only shrinks.

    Mixing two unit-variance signals with convex weights gives variance at most
    one, with equality only at perfect correlation -- so the mixture is always
    flatter than the routed head alone. That shrinkage is precisely what made
    Loss regress while accuracy improved, and it is what the sharpen factor
    exists to offset. Asserting it keeps the explanation honest.
    """
    routed, rp = _two_heads()
    args_ref[0] = SimpleNamespace(
        rp_class_fusion_gate='margin', rp_class_fusion_sharpen=1.0,
        rp_fusion_ramp=0.0, num_tasks=10)
    mixed = fuse_class_scores(routed, rp, 0.5, seen_tasks=10)
    valid = torch.isfinite(routed)
    routed_mean, routed_std = _valid_moments(routed, valid)
    mixed_mean, mixed_std = _valid_moments(mixed, valid)
    assert torch.allclose(routed_mean, mixed_mean, atol=1e-3)
    assert torch.all(mixed_std <= routed_std * 1.001)
    # Loose floor: this only has to catch the mixture collapsing, and the
    # theoretical worst case for two independent unit signals at share 0.5 is
    # 0.707 before per-sample sampling noise over 40 classes.
    assert torch.all(mixed_std > routed_std * 0.3)


def test_class_fusion_keeps_masked_classes_masked():
    routed, rp = _two_heads()
    routed[:, -5:] = float('-inf')
    args_ref[0] = SimpleNamespace(
        rp_class_fusion_gate='margin', rp_class_fusion_sharpen=1.0,
        rp_fusion_ramp=0.0, num_tasks=10)
    mixed = fuse_class_scores(routed, rp, 0.5, seen_tasks=10)
    assert torch.isinf(mixed[:, -5:]).all() and (mixed[:, -5:] < 0).all()
    assert torch.isfinite(mixed[:, :-5]).all()


def test_sharpening_is_monotone_and_cannot_change_accuracy():
    """Sharpening scales per sample, so the ranking must be bit-identical."""
    routed, rp = _two_heads()
    base = SimpleNamespace(
        rp_class_fusion_gate='margin', rp_class_fusion_sharpen=1.0,
        rp_fusion_ramp=0.0, num_tasks=10)
    args_ref[0] = base
    soft = fuse_class_scores(routed, rp, 0.5, seen_tasks=10).argmax(dim=1)
    base.rp_class_fusion_sharpen = 3.0
    sharp = fuse_class_scores(routed, rp, 0.5, seen_tasks=10).argmax(dim=1)
    assert torch.equal(soft, sharp)


# --------------------------------------------------------------------------
# Helpers the gate is built on
# --------------------------------------------------------------------------

def test_top2_margin_depends_only_on_the_ratio_of_the_top_two():
    """The docstring's claim: the quantity is (1 - r) / (1 + r) with r = p2/p1."""
    valid = torch.ones(1, 4, dtype=torch.bool)
    # Same top two either way, but the pair holds nearly all the mass in the
    # first row and well under half of it in the second. Dividing by p1 + p2
    # cancels that difference, which is the point of the definition.
    a = torch.tensor([[2.0, 1.0, -20.0, -20.0]])
    b = torch.tensor([[2.0, 1.0, 1.0, 1.0]])
    assert torch.allclose(_top2_margin(a, valid), _top2_margin(b, valid),
                          atol=1e-6)


def test_top2_margin_is_zero_for_a_tie_and_one_for_a_landslide():
    valid = torch.ones(2, 3, dtype=torch.bool)
    scores = torch.tensor([[1.0, 1.0, 0.0], [1e3, 0.0, 0.0]])
    margin = _top2_margin(scores, valid)
    assert float(margin[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(margin[1]) == pytest.approx(1.0, abs=1e-6)


def test_standardize_ignores_masked_classes():
    scores = torch.tensor([[1.0, 3.0, 999.0]])
    valid = torch.tensor([[True, True, False]])
    out = _standardize_valid(scores, valid)
    assert float(out[0, 2]) == 0.0
    assert float(out[0, :2].mean()) == pytest.approx(0.0, abs=1e-6)


def test_stage_ramp_identities():
    assert _stage_ramp(3, 10, 0.0) == 1.0        # gamma = 0 is the constant
    assert _stage_ramp(10, 10, 2.0) == 1.0       # full weight at the last stage
    assert 0.0 < _stage_ramp(5, 10, 2.0) < 1.0


# --------------------------------------------------------------------------
# Router fusion
# --------------------------------------------------------------------------

def _route_args(weight, ls_weight=0.0):
    return SimpleNamespace(
        rp_route_fusion_weight=weight, rp_fusion_ramp_scope='class',
        rp_fusion_ramp=0.0, num_tasks=10,
        rp_route_fusion_ls_weight=ls_weight)


def test_full_weight_on_tii_reproduces_tii_routing():
    """w = 1.0 must hand routing entirely to TII."""
    torch.manual_seed(0)
    class_mask = [[0, 1], [2, 3], [4, 5]]
    tii = torch.randn(64, 6) * 4.0
    rp = torch.randn(64, 6) * 0.1
    routed = fuse_routers(rp, tii, class_mask, 3, _route_args(1.0), DEVICE)
    expected = torch.stack(
        [tii[:, m].max(dim=1).values for m in class_mask], dim=1).argmax(dim=1)
    assert torch.equal(routed, expected)


def test_router_fusion_is_finite_at_the_first_stage():
    """One task means one column; the std must not divide by n - 1 = 0.

    The NaN this guards against was invisible because argmax over a single
    column returns 0 whatever the column holds -- the right answer for the
    wrong reason.
    """
    torch.manual_seed(0)
    class_mask = [[0, 1, 2]]
    tii = torch.randn(32, 3) * 4.0
    rp = torch.randn(32, 3) * 0.1
    routed = fuse_routers(rp, tii, class_mask, 1, _route_args(0.7), DEVICE)
    assert torch.equal(routed, torch.zeros(32, dtype=routed.dtype))


def test_router_fusion_handles_masked_unseen_classes():
    torch.manual_seed(0)
    class_mask = [[0, 1], [2, 3]]
    tii = torch.randn(16, 6) * 4.0
    rp = torch.randn(16, 6) * 0.1
    rp[:, 4:] = float('-inf')
    tii[:, 4:] = float('-inf')
    routed = fuse_routers(rp, tii, class_mask, 2, _route_args(0.7), DEVICE)
    assert torch.isfinite(routed.float()).all()
    assert int(routed.max()) < 2


# --------------------------------------------------------------------------
# RP head
# --------------------------------------------------------------------------

def test_reset_clears_every_field_including_the_class_mask():
    reset_rp_head()
    args = _rp_args()
    features = torch.randn(32, 16)
    targets = torch.randint(0, 4, (32,))
    accumulate_rp_statistics(features, targets, args, DEVICE, 4)
    solve_rp_head(args, DEVICE, seen_class_ids=[0, 1])
    state = get_rp_state()
    assert state['class_mask_bias'] is not None
    reset_rp_head()
    assert not rp_head_ready()
    for key in ('projection', 'gram', 'prototypes', 'weights',
                'class_mask_bias'):
        assert state.get(key) is None, '%s survived the reset' % key


def test_ridge_solve_matches_the_closed_form():
    """Cholesky path must equal (G + lambda I)^-1 C computed independently."""
    reset_rp_head()
    lam = 1e4
    args = _rp_args(rp_dim=32, rp_lambda=lam)
    torch.manual_seed(0)
    features = torch.randn(128, 16)
    targets = torch.randint(0, 5, (128,))
    accumulate_rp_statistics(features, targets, args, DEVICE, 5)
    state = get_rp_state()
    gram, prototypes = state['gram'].clone(), state['prototypes'].clone()
    weights = solve_rp_head(args, DEVICE)
    expected = torch.linalg.solve(
        gram + lam * torch.eye(gram.shape[0], dtype=torch.float64),
        prototypes).float()
    assert torch.allclose(weights, expected, atol=1e-5, rtol=1e-3)


def test_zero_dim_skips_the_projection():
    """rp_dim <= 0 is the paper's no-RP reference point: plain ridge."""
    args = _rp_args(rp_dim=0)
    features = torch.randn(8, 16)
    assert torch.equal(project_features(features, args, DEVICE), features)


def test_projection_is_deterministic_from_the_seed():
    reset_rp_head()
    args = _rp_args(rp_dim=32, rp_seed=7)
    features = torch.randn(8, 16)
    first = project_features(features, args, DEVICE).clone()
    reset_rp_head()
    second = project_features(features, args, DEVICE)
    assert torch.equal(first, second)


def test_unseen_classes_are_masked_to_negative_infinity():
    reset_rp_head()
    args = _rp_args(rp_dim=32)
    torch.manual_seed(0)
    features = torch.randn(64, 16)
    targets = torch.randint(0, 6, (64,))
    accumulate_rp_statistics(features, targets, args, DEVICE, 6)
    solve_rp_head(args, DEVICE, seen_class_ids=[0, 1, 2])
    scores = rp_head_predict(torch.randn(8, 16), args, DEVICE)
    assert torch.isfinite(scores[:, :3]).all()
    assert torch.isinf(scores[:, 3:]).all() and (scores[:, 3:] < 0).all()


def test_predict_before_solve_raises():
    reset_rp_head()
    args = _rp_args()
    with pytest.raises(RuntimeError, match='not solved'):
        rp_head_predict(torch.randn(4, 16), args, DEVICE)


def test_accumulation_is_order_independent():
    """Gram and prototypes are sums, so shuffling the batches cannot move them.

    This is the property that makes the head immune to catastrophic forgetting,
    and the method section leans on it.
    """
    args = _rp_args(rp_dim=32)
    torch.manual_seed(0)
    features = torch.randn(96, 16)
    targets = torch.randint(0, 5, (96,))

    reset_rp_head()
    for start in range(0, 96, 32):
        accumulate_rp_statistics(features[start:start + 32],
                                 targets[start:start + 32], args, DEVICE, 5)
    forward = solve_rp_head(args, DEVICE).clone()

    reset_rp_head()
    for start in reversed(range(0, 96, 32)):
        accumulate_rp_statistics(features[start:start + 32],
                                 targets[start:start + 32], args, DEVICE, 5)
    backward = solve_rp_head(args, DEVICE)
    assert torch.allclose(forward, backward, atol=1e-5)
