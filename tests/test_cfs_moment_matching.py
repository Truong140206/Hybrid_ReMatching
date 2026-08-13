import torch

from utils import match_cfs_feature_moments


def test_cfs_moment_matching_restores_target_mean_and_standard_deviation():
    selected = torch.tensor([
        [-4.0, 1.0, 10.0],
        [-1.0, 2.0, 14.0],
        [3.0, 5.0, 18.0],
        [8.0, 9.0, 30.0],
    ])
    target_mean = torch.tensor([2.0, -3.0, 5.0])
    target_variance = torch.tensor([4.0, 9.0, 16.0])

    matched = match_cfs_feature_moments(
        selected, target_mean, torch.diag(target_variance))

    assert torch.allclose(matched.mean(dim=0), target_mean, atol=1e-6)
    assert torch.allclose(
        matched.std(dim=0, unbiased=True), target_variance.sqrt(), atol=1e-6)


def test_cfs_moment_matching_leaves_single_sample_unchanged():
    selected = torch.tensor([[1.0, 2.0]])

    matched = match_cfs_feature_moments(
        selected, torch.zeros(2), torch.eye(2))

    assert torch.equal(matched, selected)
