from types import SimpleNamespace

import torch
from torch import nn

from engines.progressive_rematching import progressive_adapter_rematching


class TaskAwareModel(nn.Module):
    def forward(self, inputs, task_id):
        logits = torch.full((inputs.shape[0], 6), -5.0, device=inputs.device)
        for row, selected_task in enumerate(task_id.tolist()):
            start = 2 * selected_task
            logits[row, start] = inputs[row, selected_task]
            logits[row, start + 1] = inputs[row, selected_task] - 1.0
        return {'logits': logits}


def _args(**overrides):
    values = dict(
        progressive_logit_temperature=1.0,
        progressive_tii_prior_weight=0.0,
        progressive_initial_candidates=2,
        progressive_intermediate_candidates=2,
        progressive_stage1_tii_margin=0.5,
        progressive_stage1_adapter_margin=0.5,
        progressive_stage1_class_margin=0.5,
        progressive_stage2_tii_margin=0.0,
        progressive_stage2_adapter_margin=0.0,
        progressive_stage2_class_margin=0.0,
        progressive_excluded_logit_margin=20.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_confident_sample_stops_after_initial_candidates():
    inputs = torch.tensor([[5.0, 2.0, 1.0]])
    tii_logits = torch.tensor([[5.0, 4.0, 2.0, 1.0, 0.0, -1.0]])

    logits, routed, counts, stages = progressive_adapter_rematching(
        TaskAwareModel(), inputs, tii_logits,
        class_mask=[[0, 1], [2, 3], [4, 5]],
        seen_task_count=3, args=_args())

    assert routed.tolist() == [0]
    assert counts.tolist() == [2.0]
    assert stages.tolist() == [1]
    assert logits.argmax(dim=1).tolist() == [0]


def test_uncertain_sample_falls_back_to_all_adapters():
    inputs = torch.tensor([[3.0, 2.9, 6.0]])
    tii_logits = torch.tensor([[3.0, 2.9, 2.8, 2.7, 2.6, 2.5]])

    logits, routed, counts, stages = progressive_adapter_rematching(
        TaskAwareModel(), inputs, tii_logits,
        class_mask=[[0, 1], [2, 3], [4, 5]],
        seen_task_count=3,
        args=_args(progressive_stage1_tii_margin=10.0))

    assert routed.tolist() == [2]
    assert counts.tolist() == [3.0]
    assert stages.tolist() == [3]
    assert logits.argmax(dim=1).tolist() == [4]
