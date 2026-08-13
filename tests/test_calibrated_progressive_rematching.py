import torch

from engines.calibrated_progressive_rematching import (
    _choose_precision_threshold,
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
