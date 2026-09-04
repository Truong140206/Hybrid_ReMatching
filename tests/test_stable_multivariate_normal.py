"""A singular class covariance must not kill the run.

ImageNet-A has classes holding a few dozen images. The empirical covariance of
768-dimensional features from that many samples is rank-deficient, and
MultivariateNormal rejects it, which is what stopped ImageNet-A training on
three of the five backbones after three tasks each.

Two properties matter and both are checked here: a covariance that is already
positive definite must come back untouched, so nothing already measured
changes; and a singular one must come back usable rather than raising.
"""
import torch

import utils


def _singular_covariance(dim=32, rank=4):
    """Empirical covariance of fewer samples than dimensions -- always singular."""
    torch.manual_seed(0)
    samples = torch.randn(rank, dim)
    centred = samples - samples.mean(dim=0, keepdim=True)
    return centred.t().mm(centred) / rank


def test_positive_definite_covariance_is_left_alone():
    dim = 16
    cov = torch.eye(dim) * 2.5
    mean = torch.zeros(dim)

    guarded = utils.stable_multivariate_normal(mean, cov, 'test-pd')
    plain = torch.distributions.MultivariateNormal(mean.float(), cov.float())

    assert torch.equal(guarded.covariance_matrix, plain.covariance_matrix)
    assert torch.equal(guarded.loc, plain.loc)


def test_singular_covariance_becomes_usable():
    cov = _singular_covariance()
    mean = torch.zeros(cov.shape[0])

    # The unguarded construction is the one that killed the run.
    raised = False
    try:
        torch.distributions.MultivariateNormal(mean, cov)
    except (ValueError, RuntimeError):
        raised = True
    assert raised, 'covariance nay dang le phai suy bien'

    distribution = utils.stable_multivariate_normal(mean, cov, 'test-singular')
    drawn = distribution.sample(sample_shape=(8,))
    assert drawn.shape == (8, cov.shape[0])
    assert torch.isfinite(drawn).all()


def test_jitter_is_reported_once_per_label():
    cov = _singular_covariance()
    mean = torch.zeros(cov.shape[0])
    utils._JITTERED.discard('test-once')                    # noqa: SLF001

    utils.stable_multivariate_normal(mean, cov, 'test-once')
    assert 'test-once' in utils._JITTERED                   # noqa: SLF001
    # Second call must not re-announce; it simply returns.
    utils.stable_multivariate_normal(mean, cov, 'test-once')
