from types import SimpleNamespace

import torch

from protocols import validate_exemplar_free_protocol
from engines.progressive_oracle_audit import prediction_proposal_diagnostics


def _audit(initial_logits):
    tii_ranking = torch.tensor([[0, 1, 2, 3, 4]])
    rank_logits = torch.full((1, 5, 10), float('-inf'))
    rank_logits[0, 0, 0] = 5.0
    rank_logits[0, 1, 2] = 4.0
    rank_logits[0, 2, 4] = 3.0
    rank_logits[0, 3, 6] = 2.0
    rank_logits[0, 4, 8] = 8.0
    task_evidence = torch.tensor([[5.0, 4.0, 3.0, 2.0, 8.0]])
    return prediction_proposal_diagnostics(
        tii_ranking=tii_ranking,
        initial_adapter_logits=initial_logits,
        rank_logits=rank_logits,
        task_evidence=task_evidence,
        winner_tasks=torch.tensor([4]),
        full_predictions=torch.tensor([8]),
        class_mask=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        initial_count=2,
        proposal_count=2,
        top_classes=2,
    )


def test_prediction_proposal_recovers_late_exhaustive_winner():
    initial_logits = torch.full((1, 2, 10), -10.0)
    initial_logits[0, 0, 8] = 9.0
    initial_logits[0, 0, 4] = 8.0
    initial_logits[0, 1, 9] = 7.0
    initial_logits[0, 1, 5] = 6.0

    audit = _audit(initial_logits)

    assert audit['prediction_proposal_winner_recall'].tolist() == [True]
    assert audit['prediction_proposal_exact_agreement'].tolist() == [True]
    assert audit['prediction_proposal_lora_counts'].tolist() == [4.0]
    assert audit['prediction_proposal_new_winner'].tolist() == [True]


def test_prediction_proposal_falls_back_to_tii_when_no_new_task_is_predicted():
    initial_logits = torch.full((1, 2, 10), -10.0)
    initial_logits[0, 0, 0] = 9.0
    initial_logits[0, 0, 1] = 8.0
    initial_logits[0, 1, 2] = 7.0
    initial_logits[0, 1, 3] = 6.0

    audit = _audit(initial_logits)

    assert audit['prediction_proposal_winner_recall'].tolist() == [False]
    assert audit['prediction_proposal_exact_agreement'].tolist() == [False]
    assert audit['prediction_proposal_lora_counts'].tolist() == [4.0]
    assert audit['prediction_proposal_new_winner'].tolist() == [False]


def test_prediction_proposal_is_allowed_by_strict_exemplar_free_protocol():
    validate_exemplar_free_protocol(SimpleNamespace(
        strict_exemplar_free=True,
        progressive_oracle_audit=True,
        progressive_prediction_proposal_audit=True,
    ))