import torch

from engines.calibrated_progressive_rematching import (
    HaltingGate,
    _cascade_stage2_mask,
    _choose_output_temperature,
    _choose_precision_threshold,
    _finalize_partial_logits,
    _stage_features,
)


def test_precision_threshold_prefers_largest_safe_coverage():
    probabilities = torch.tensor([0.99, 0.95, 0.80, 0.70])
    labels = torch.tensor([1, 1, 0, 1], dtype=torch.bool)

    threshold, precision, coverage = _choose_precision_threshold(
        probabilities, labels, target_precision=1.0, minimum_coverage=0.25)

    assert abs(threshold - 0.95) < 1e-6
    assert precision == 1.0
    assert coverage == 0.5


def test_stage_features_measure_unseen_tii_margin():
    tii_prior = torch.tensor([[2.0, 1.0, 0.5]])
    sorted_tii = torch.tensor([[2.0, 1.0, 0.5]])
    evidence = torch.tensor([[4.0, 2.0, float('-inf')]])
    class_margins = torch.tensor([[1.5, 0.5, 0.0]])

    features = _stage_features(
        tii_prior, sorted_tii, evidence, class_margins, boundary=2)

    assert features.shape == (1, 8)
    assert torch.isclose(features[0, 1], torch.tensor(2.0))
    assert torch.isclose(features[0, 2], torch.tensor(1.5))
    assert torch.isclose(features[0, 4], torch.tensor(1.5))


def test_stage2_mask_uses_stage1_residual_and_keeps_each_class_trainable():
    gate = HaltingGate(feature_dim=1, hidden_dim=2, dropout=0.0)
    gate.network = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        gate.network.weight.fill_(1.0)
    gate.register_buffer('feature_mean', torch.zeros(1, 1))
    gate.register_buffer('feature_std', torch.ones(1, 1))
    gate.threshold = 0.5
    features = torch.tensor([[-2.0], [1.0], [2.0], [-3.0], [-1.0], [3.0]])
    class_targets = torch.tensor([0, 0, 0, 1, 1, 1])

    selected = _cascade_stage2_mask(
        gate, features, class_targets, minimum_per_class=2, context_ratio=0.0)

    assert selected.tolist() == [True, True, False, True, True, False]


def test_partial_logit_margin_preserves_predictions_and_caps_penalty():
    logits = torch.tensor([
        [3.0, 2.0, float('-inf'), float('-inf')],
        [1.0, 4.0, float('-inf'), float('-inf')],
    ])

    finalized = _finalize_partial_logits(logits, excluded_margin=8.0)

    assert torch.equal(finalized.argmax(1), torch.tensor([0, 1]))
    assert torch.equal(
        torch.topk(finalized, k=2, dim=1).indices,
        torch.topk(logits, k=2, dim=1).indices)
    assert torch.allclose(finalized[:, 2:], torch.tensor([
        [-6.0, -6.0],
        [-7.0, -7.0],
    ]))


def test_stage2_mask_adds_near_boundary_context():
    gate = HaltingGate(feature_dim=1, hidden_dim=2, dropout=0.0)
    gate.network = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        gate.network.weight.fill_(1.0)
    gate.register_buffer('feature_mean', torch.zeros(1, 1))
    gate.register_buffer('feature_std', torch.ones(1, 1))
    gate.threshold = 0.5
    features = torch.tensor([[-2.0], [0.1], [0.5], [1.0], [2.0]])
    class_targets = torch.zeros(5, dtype=torch.long)

    selected = _cascade_stage2_mask(
        gate, features, class_targets,
        minimum_per_class=2, context_ratio=0.5)

    assert selected.tolist() == [True, True, True, True, False]


def test_output_temperature_reduces_nll_without_changing_predictions():
    logits = torch.tensor([
        [8.0, 0.0],
        [8.0, 0.0],
        [8.0, 0.0],
    ])
    targets = torch.tensor([0, 0, 1])

    temperature, before, after = _choose_output_temperature(
        logits, targets, minimum=0.5, maximum=4.0, steps=65)

    assert temperature > 1.0
    assert after < before
    assert torch.equal(
        logits.argmax(1), (logits / temperature).argmax(1))
