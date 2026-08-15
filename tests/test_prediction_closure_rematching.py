from types import SimpleNamespace

import torch
from torch import nn

from engines.prediction_closure_rematching import (
    prediction_closure_tii_tail_rematching,
)
from engines.progressive_oracle_audit import progressive_oracle_audit


def _args():
    return SimpleNamespace(
        progressive_logit_temperature=1.0,
        progressive_tii_prior_weight=0.0,
        progressive_excluded_logit_margin=20.0,
        prediction_proposal_initial_count=2,
        prediction_proposal_top_classes=1,
        progressive_prediction_closure_audit=True,
        progressive_prediction_closure_tii_tail_audit=True,
        progressive_prediction_proposal_audit=False,
        progressive_arrow_audit=False,
        progressive_lora_response_audit=False,
    )


class ChainClosureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, inputs, task_id):
        self.batch_sizes.append(int(inputs.shape[0]))
        logits = torch.full(
            (inputs.shape[0], 4), -10.0, device=inputs.device)
        for row, selected_task in enumerate(task_id.tolist()):
            if selected_task == 0:
                logits[row, 0] = 2.0
                logits[row, 2] = 9.0
            elif selected_task == 1:
                logits[row, 1] = 1.0
            elif selected_task == 2:
                logits[row, 2] = 3.0
                logits[row, 3] = 9.0
            else:
                logits[row, 3] = 5.0
        return {'logits': logits}


def test_operational_closure_matches_oracle_projection_exactly():
    inputs = torch.tensor([[0.0]])
    tii_logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    class_mask = [[0], [1], [2], [3]]
    args = _args()

    operational_model = ChainClosureModel()
    output, routed, diagnostics = prediction_closure_tii_tail_rematching(
        operational_model, inputs, tii_logits, class_mask, 4, args)

    oracle_model = ChainClosureModel()
    oracle_output, oracle_routed, oracle = progressive_oracle_audit(
        oracle_model, inputs, tii_logits, class_mask, 4, args)

    assert torch.allclose(output, oracle_output, atol=1e-6)
    assert routed.tolist() == oracle_routed.tolist() == [3]
    assert diagnostics['lora_counts'].tolist() == [4.0]
    assert diagnostics['forward_calls'].tolist() == [3.0]
    assert diagnostics['lora_counts'].tolist() == oracle[
        'prediction_closure_lora_counts'].tolist()
    assert diagnostics['forward_calls'].tolist() == oracle[
        'prediction_closure_forward_calls'].tolist()
    assert operational_model.batch_sizes == [2, 1, 1]
    assert oracle_model.batch_sizes == [1, 1, 1, 1]


class PerSampleClosureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, inputs, task_id):
        self.batch_sizes.append(int(inputs.shape[0]))
        logits = torch.full(
            (inputs.shape[0], 3), -10.0, device=inputs.device)
        for row, (sample_id, selected_task) in enumerate(zip(
                inputs[:, 0].tolist(), task_id.tolist())):
            if selected_task == 0:
                logits[row, 0] = 5.0
                if sample_id == 0.0:
                    logits[row, 2] = 9.0
            elif selected_task == 1:
                logits[row, 1] = 4.0
            else:
                logits[row, 2] = 3.0
        return {'logits': logits}


def test_operational_closure_batches_only_active_sample_task_pairs():
    model = PerSampleClosureModel()
    args = _args()
    output, routed, diagnostics = prediction_closure_tii_tail_rematching(
        model=model,
        inputs=torch.tensor([[0.0], [1.0]]),
        tii_logits=torch.tensor([[3.0, 2.0, 1.0],
                                 [3.0, 2.0, 1.0]]),
        class_mask=[[0], [1], [2]],
        seen_task_count=3,
        args=args,
    )

    assert torch.isfinite(output).all()
    assert routed.tolist() == [0, 0]
    assert diagnostics['lora_counts'].tolist() == [3.0, 2.0]
    assert diagnostics['forward_calls'].tolist() == [2.0, 1.0]
    assert model.batch_sizes == [4, 1]
