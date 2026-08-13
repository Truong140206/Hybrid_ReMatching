from types import SimpleNamespace

import pytest

from protocols import validate_exemplar_free_protocol


def test_strict_protocol_accepts_cfs_statistics_and_synthetic_replay():
    args = SimpleNamespace(
        strict_exemplar_free=True,
        cfs_sampling=True,
        replay_anchor_ctird=True,
    )
    validate_exemplar_free_protocol(args)


@pytest.mark.parametrize('flag', [
    'crct_real_feature_replay',
    'prototype_rematching',
    'shared_prototype_router',
    'replay_logit_calibration',
    'replay_task_router',
    'calibrated_progressive_rematching',
    'distilled_router_rematching',
])
def test_strict_protocol_rejects_historical_data_features(flag):
    args = SimpleNamespace(strict_exemplar_free=True, **{flag: True})
    with pytest.raises(ValueError, match=flag):
        validate_exemplar_free_protocol(args)


def test_strict_protocol_rejects_local_real_prototypes():
    args = SimpleNamespace(
        strict_exemplar_free=True,
        exhaustive_local_prototype_weight=0.1,
    )
    with pytest.raises(ValueError, match='exhaustive_local_prototype_weight'):
        validate_exemplar_free_protocol(args)
