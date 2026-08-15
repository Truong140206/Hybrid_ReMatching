from types import SimpleNamespace

import pytest
import torch
from torch import nn

from engines.cfs_task_logit_calibration import (
    build_cfs_synthetic_feature_memory,
    validate_cfs_task_logit_calibration_state,
)
from engines.prediction_proposal_rematching import (
    prediction_proposal_adapter_rematching,
)
from engines.replay_logit_calibration import calibrate_task_logits


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(2, 4, bias=True)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]))
            self.head.bias.zero_()

    def forward(self, features, fc_only=False):
        return {'logits': self.head(self.fc_norm(features))}


def _calibration_args():
    return SimpleNamespace(
        replay_calibration_steps=5,
        replay_calibration_lr=0.05,
        replay_calibration_max_scale=1.25,
        replay_calibration_max_bias=0.5,
        replay_calibration_regularization=0.05,
        replay_calibration_old_margin_weight=0.5,
        replay_calibration_old_tolerance=0.0,
        replay_calibration_min_gain=0.0,
        replay_calibration_max_old_class_drop=0.0,
        replay_calibration_batch_size=32,
    )


def _feature_memory(offset=0.0):
    return {
        0: torch.tensor([[2.0 + offset, 0.0]]).repeat(4, 1),
        1: torch.tensor([[0.0, 2.0 + offset]]).repeat(4, 1),
        2: torch.tensor([[-2.0 - offset, 0.0]]).repeat(4, 1),
        3: torch.tensor([[0.0, -2.0 - offset]]).repeat(4, 1),
    }


def test_calibration_uses_independent_report_and_does_not_mutate_head():
    model = TinyClassifier()
    weight_before = model.head.weight.detach().clone()
    bias_before = model.head.bias.detach().clone()

    result = calibrate_task_logits(
        model=model,
        feature_memory=_feature_memory(),
        report_feature_memory=_feature_memory(offset=0.1),
        class_mask=[[0, 1], [2, 3]],
        seen_task_count=2,
        args=_calibration_args(),
        device=torch.device('cpu'),
        apply_to_model=False,
        max_samples_per_class=4,
    )

    assert result['samples'] == 16
    assert result['report_samples'] == 16
    assert result['applied_to_model'] is False
    assert len(result['scale']) == 2
    assert len(result['bias']) == 2
    assert torch.equal(model.head.weight, weight_before)
    assert torch.equal(model.head.bias, bias_before)


def test_cfs_synthetic_memory_contains_only_requested_transient_features():
    args = SimpleNamespace(
        cfs_paper_style=True,
        cfs_selection_ratio=0.5,
        cfs_selection_steps=2,
        cfs_candidate_multiplier=2,
        cfs_step_candidates=0,
        cfs_tau=1.0,
        cfs_distribution_filter=False,
        cfs_moment_match=True,
    )
    memory = build_cfs_synthetic_feature_memory(
        cls_mean={0: torch.tensor([1.0, 0.0]),
                  1: torch.tensor([0.0, 1.0])},
        cls_cov={0: torch.tensor([0.2, 0.1]),
                 1: torch.tensor([0.1, 0.2])},
        cls_cfs_model={0: nn.Identity(), 1: nn.Identity()},
        class_mask=[[0, 1]],
        seen_task_count=1,
        samples_per_class=6,
        args=args,
        device=torch.device('cpu'),
    )

    assert set(memory) == {0, 1}
    assert memory[0].shape == (6, 2)
    assert memory[1].shape == (6, 2)
    assert memory[0].device.type == 'cpu'
    assert memory[1].device.type == 'cpu'


def _valid_state():
    return {
        'version': 1,
        'source': 'cfs_aggregate_statistics',
        'accepted': True,
        'reason': 'synthetic_report_gate_pass',
        'scale': [1.0, 1.1],
        'bias': [-0.1, 0.1],
        'fit_samples': 16,
        'report_samples': 16,
        'before': {'all': 80.0, 'old': 81.0, 'new': 79.0},
        'after': {'all': 81.0, 'old': 81.0, 'new': 81.0},
        'before_loss': 0.5,
        'after_loss': 0.4,
        'worst_old_class_drop': 0.0,
    }


def test_calibration_state_validator_rejects_feature_payloads():
    state = _valid_state()
    assert validate_cfs_task_logit_calibration_state(state, 2)

    state['features'] = torch.ones(2, 2)
    with pytest.raises(ValueError, match='unexpected payloads'):
        validate_cfs_task_logit_calibration_state(state, 2)


class TwoTaskProposalModel(nn.Module):
    def forward(self, inputs, task_id):
        logits = torch.full((inputs.shape[0], 4), -10.0)
        for row, selected_task in enumerate(task_id.tolist()):
            if selected_task == 0:
                logits[row, 0] = 5.0
            else:
                logits[row, 2] = 4.0
        return {'logits': logits}


def test_prediction_proposal_applies_task_calibration_before_routing():
    args = SimpleNamespace(
        progressive_logit_temperature=1.0,
        progressive_tii_prior_weight=0.0,
        progressive_excluded_logit_margin=20.0,
        prediction_proposal_initial_count=2,
        prediction_proposal_count=0,
        prediction_proposal_top_classes=1,
        cfs_task_logit_calibration=True,
        cfs_task_logit_calibration_state={
            'scale': [1.0, 1.0],
            'bias': [0.0, 2.0],
        },
    )

    logits, routed, _ = prediction_proposal_adapter_rematching(
        model=TwoTaskProposalModel(),
        inputs=torch.tensor([[1.0]]),
        tii_logits=torch.tensor([[2.0, 1.0, 0.0, -1.0]]),
        class_mask=[[0, 1], [2, 3]],
        seen_task_count=2,
        args=args,
    )

    assert routed.tolist() == [1]
    assert logits.argmax(dim=1).tolist() == [2]


def test_prediction_proposal_fails_closed_without_calibration_state():
    args = SimpleNamespace(
        progressive_logit_temperature=1.0,
        progressive_tii_prior_weight=0.0,
        progressive_excluded_logit_margin=20.0,
        prediction_proposal_initial_count=2,
        prediction_proposal_count=0,
        prediction_proposal_top_classes=1,
        cfs_task_logit_calibration=True,
    )

    with pytest.raises(RuntimeError, match='no checkpoint state'):
        prediction_proposal_adapter_rematching(
            model=TwoTaskProposalModel(),
            inputs=torch.tensor([[1.0]]),
            tii_logits=torch.tensor([[2.0, 1.0, 0.0, -1.0]]),
            class_mask=[[0, 1], [2, 3]],
            seen_task_count=2,
            args=args,
        )