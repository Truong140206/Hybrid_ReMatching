from types import SimpleNamespace

import torch
from torch import nn

from engines.progressive_oracle_audit import (
    progressive_oracle_audit,
    stage_drift_diagnostics,
)


class TaskAwareModel(nn.Module):
    def forward(self, inputs, task_id):
        logits = torch.full((inputs.shape[0], 6), -5.0, device=inputs.device)
        for row, selected_task in enumerate(task_id.tolist()):
            start = 2 * selected_task
            logits[row, start] = inputs[row, selected_task]
            logits[row, start + 1] = inputs[row, selected_task] - 1.0
        return {'logits': logits}


def _args():
    return SimpleNamespace(
        progressive_logit_temperature=1.0,
        progressive_tii_prior_weight=0.0,
        progressive_excluded_logit_margin=20.0,
    )


def test_oracle_audit_finds_late_exhaustive_winner():
    inputs = torch.tensor([[3.0, 2.9, 6.0]])
    tii_logits = torch.tensor([[3.0, 2.9, 2.8, 2.7, 2.6, 2.5]])

    logits, routed, audit = progressive_oracle_audit(
        TaskAwareModel(), inputs, tii_logits,
        class_mask=[[0, 1], [2, 3], [4, 5]],
        seen_task_count=3, args=_args())

    assert routed.tolist() == [2]
    assert logits.argmax(dim=1).tolist() == [4]
    assert audit['winner_recall_2'].tolist() == [False]
    assert audit['exact_agreement_2'].tolist() == [False]
    assert audit['oracle_lora_counts'].tolist() == [3.0]
    assert audit['actual_lora_counts'].tolist() == [3.0]


def test_oracle_audit_stops_easy_sample_at_two():
    inputs = torch.tensor([[5.0, 2.0, 1.0]])
    tii_logits = torch.tensor([[5.0, 4.0, 2.0, 1.0, 0.0, -1.0]])

    _, routed, audit = progressive_oracle_audit(
        TaskAwareModel(), inputs, tii_logits,
        class_mask=[[0, 1], [2, 3], [4, 5]],
        seen_task_count=3, args=_args())

    assert routed.tolist() == [0]
    assert audit['winner_recall_2'].tolist() == [True]
    assert audit['exact_agreement_2'].tolist() == [True]
    assert audit['oracle_lora_counts'].tolist() == [2.0]

def test_stage_drift_separates_local_accuracy_from_seen_competition():
    candidate_tasks = torch.tensor([[0, 1], [1, 0]])
    full_adapter_logits = torch.tensor([
        [
            [5.0, 4.0, 6.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        [
            [0.0, 0.0, 1.0, 0.0],
            [4.0, 5.0, 3.0, 2.0],
        ],
    ])
    diagnostics = stage_drift_diagnostics(
        candidate_tasks=candidate_tasks,
        full_adapter_logits=full_adapter_logits,
        targets=torch.tensor([0, 1]),
        true_task=0,
        class_mask=[[0, 1], [2, 3]],
        seen_task_count=2,
    )

    assert diagnostics['stage_drift_own_local_correct'].tolist() == [
        True, True]
    assert diagnostics['stage_drift_own_seen_correct'].tolist() == [
        False, True]
    assert diagnostics['stage_drift_own_seen_task_correct'].tolist() == [
        False, True]
    assert diagnostics['stage_drift_local_to_seen_failure'].tolist() == [
        True, False]
    assert torch.isfinite(
        diagnostics['stage_drift_own_local_loss']).all()
    assert torch.isfinite(
        diagnostics['stage_drift_own_seen_loss']).all()
