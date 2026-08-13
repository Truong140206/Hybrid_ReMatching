from types import SimpleNamespace

import torch

from engines.calibrated_progressive_rematching import (
    HaltingGate,
    _apply_rank_preserving_smoothing,
    _cascade_stage2_mask,
    _choose_output_temperature,
    _choose_precision_threshold,
    _finalize_partial_logits,
    _run_lora_rank_batch,
    calibrated_progressive_rematching,
    _stage_features,
)


class _TaskAwareDummy(torch.nn.Module):
    def forward(self, inputs, task_id):
        task_id = task_id.to(inputs.dtype).unsqueeze(1)
        class_scale = torch.arange(
            1, inputs.shape[1] + 1, dtype=inputs.dtype, device=inputs.device)
        return {'logits': inputs + task_id * class_scale.unsqueeze(0)}


def test_rank_batch_matches_serial_lora_evaluation():
    model = _TaskAwareDummy()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    task_ids = torch.tensor([[0, 2], [1, 3]])

    batched = _run_lora_rank_batch(model, inputs, task_ids)
    serial = torch.stack([
        model(inputs, task_id=task_ids[:, rank])['logits']
        for rank in range(task_ids.shape[1])
    ], dim=1)

    assert torch.equal(batched, serial)


def test_progressive_rank_batch_preserves_outputs_and_halting():
    model = _TaskAwareDummy()
    inputs = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    tii_logits = torch.tensor([
        [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    ])
    class_mask = [[0, 1], [2, 3], [4, 5]]
    common = dict(
        progressive_logit_temperature=1.0,
        progressive_tii_prior_weight=0.3,
        progressive_excluded_logit_margin=8.0,
        progressive_uncertainty_smoothing=False,
    )
    serial = calibrated_progressive_rematching(
        model, inputs, tii_logits, class_mask, 3,
        SimpleNamespace(**common, progressive_lora_batch_ranks=1), {})
    batched = calibrated_progressive_rematching(
        model, inputs, tii_logits, class_mask, 3,
        SimpleNamespace(**common, progressive_lora_batch_ranks=2), {})

    for serial_value, batched_value in zip(serial[:3], batched[:3]):
        assert torch.equal(serial_value, batched_value)
    assert torch.equal(serial[4], batched[4])
    assert torch.equal(serial[3], torch.full((2,), 3.0))
    assert torch.equal(batched[3], torch.full((2,), 2.0))


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


def test_uncertainty_smoothing_preserves_ranks_and_reduces_overconfidence():
    logits = torch.tensor([
        [8.0, 0.0, -1.0],
        [8.0, 0.0, -1.0],
        [8.0, 0.0, -1.0],
    ])
    targets = torch.tensor([0, 0, 1])
    smoothing = torch.full((3,), 0.05)

    smoothed = _apply_rank_preserving_smoothing(logits, smoothing)

    assert torch.equal(logits.argmax(1), smoothed.argmax(1))
    assert torch.equal(
        torch.topk(logits, k=3, dim=1).indices,
        torch.topk(smoothed, k=3, dim=1).indices)
    assert torch.allclose(
        smoothed.exp().sum(1), torch.ones(3), atol=1e-6)
    assert torch.nn.functional.cross_entropy(
        smoothed, targets) < torch.nn.functional.cross_entropy(logits, targets)
