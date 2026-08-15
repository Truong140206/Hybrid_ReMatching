from types import SimpleNamespace

import torch
from torch import nn

from protocols import validate_exemplar_free_protocol
from engines.prediction_proposal_rematching import (
    _complete_with_tii_probability_mass,
    conditional_candidate_fusion,
    crm_confidence_candidate_fusion,
    cross_adapter_borda_consensus,
    cross_adapter_global_consensus,
    initial_branch_confidence_dominance,
    prediction_proposal_adapter_rematching,
    task_mass_preserving_candidate_fusion,
)
from engines.progressive_oracle_audit import (
    prediction_beam_closure_diagnostics,
    prediction_budget_closure_diagnostics,
    prediction_closure_diagnostics,
    prediction_majority_budget_closure_diagnostics,
    prediction_proposal_diagnostics,
)


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


def _closure_audit(full_adapter_logits, rank_logits, task_evidence,
                   winner_task, full_prediction):
    return prediction_closure_diagnostics(
        tii_ranking=torch.tensor([[0, 1, 2, 3]]),
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=task_evidence,
        winner_tasks=torch.tensor([winner_task]),
        full_predictions=torch.tensor([full_prediction]),
        class_mask=[[0], [1], [2], [3]],
        initial_count=2,
        top_classes=1,
    )


def test_prediction_closure_expands_until_prediction_fixed_point():
    full_adapter_logits = torch.full((1, 4, 4), -10.0)
    full_adapter_logits[0, 0, 2] = 9.0
    full_adapter_logits[0, 1, 1] = 8.0
    full_adapter_logits[0, 2, 3] = 9.0
    full_adapter_logits[0, 3, 3] = 10.0
    rank_logits = torch.full((1, 4, 4), float('-inf'))
    rank_logits[0, 0, 0] = 2.0
    rank_logits[0, 1, 1] = 1.0
    rank_logits[0, 2, 2] = 3.0
    rank_logits[0, 3, 3] = 5.0

    audit = _closure_audit(
        full_adapter_logits, rank_logits,
        task_evidence=torch.tensor([[2.0, 1.0, 3.0, 5.0]]),
        winner_task=3, full_prediction=3)

    assert audit['prediction_closure_winner_recall'].tolist() == [True]
    assert audit['prediction_closure_exact_agreement'].tolist() == [True]
    assert audit['prediction_closure_top5_coverage'].tolist() == [True]
    assert audit['prediction_closure_lora_counts'].tolist() == [4.0]
    assert audit['prediction_closure_forward_calls'].tolist() == [3.0]
    assert audit['prediction_closure_full_scan'].tolist() == [True]


def test_prediction_closure_stops_without_new_task_or_threshold():
    full_adapter_logits = torch.full((1, 4, 4), -10.0)
    full_adapter_logits[0, 0, 0] = 9.0
    full_adapter_logits[0, 1, 1] = 8.0
    rank_logits = torch.full((1, 4, 4), float('-inf'))
    rank_logits[0, 0, 0] = 5.0
    rank_logits[0, 1, 1] = 4.0
    rank_logits[0, 2, 2] = 3.0
    rank_logits[0, 3, 3] = 2.0

    audit = _closure_audit(
        full_adapter_logits, rank_logits,
        task_evidence=torch.tensor([[5.0, 4.0, 3.0, 2.0]]),
        winner_task=0, full_prediction=0)

    assert audit['prediction_closure_winner_recall'].tolist() == [True]
    assert audit['prediction_closure_exact_agreement'].tolist() == [True]
    assert audit['prediction_closure_top5_coverage'].tolist() == [False]
    assert audit['prediction_closure_lora_counts'].tolist() == [2.0]
    assert audit['prediction_closure_forward_calls'].tolist() == [1.0]
    assert audit['prediction_closure_full_scan'].tolist() == [False]

def test_prediction_closure_tracks_rank_and_task_masks_per_sample():
    tii_ranking = torch.tensor([[2, 0, 1], [1, 2, 0]])
    full_adapter_logits = torch.full((2, 3, 3), -10.0)
    full_adapter_logits[0, 0, 1] = 9.0
    full_adapter_logits[0, 1, 0] = 8.0
    full_adapter_logits[0, 2, 1] = 10.0
    full_adapter_logits[1, 0, 1] = 9.0
    full_adapter_logits[1, 1, 2] = 8.0
    full_adapter_logits[1, 2, 0] = 7.0
    rank_logits = torch.full((2, 3, 3), float('-inf'))
    rank_logits[0, 0, 2] = 2.0
    rank_logits[0, 1, 0] = 1.0
    rank_logits[0, 2, 1] = 5.0
    rank_logits[1, 0, 1] = 2.0
    rank_logits[1, 1, 2] = 4.0
    rank_logits[1, 2, 0] = 1.0

    audit = prediction_closure_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=torch.tensor([[2.0, 1.0, 5.0],
                                    [2.0, 4.0, 1.0]]),
        winner_tasks=torch.tensor([1, 2]),
        full_predictions=torch.tensor([1, 2]),
        class_mask=[[0], [1], [2]],
        initial_count=2,
        top_classes=1,
    )

    assert audit['prediction_closure_winner_recall'].tolist() == [True, True]
    assert audit['prediction_closure_exact_agreement'].tolist() == [True, True]
    assert audit['prediction_closure_top5_coverage'].tolist() == [True, False]
    assert audit['prediction_closure_lora_counts'].tolist() == [3.0, 2.0]
    assert audit['prediction_closure_forward_calls'].tolist() == [2.0, 1.0]


def test_prediction_closure_tii_tail_is_top1_safe_and_cost_free():
    full_adapter_logits = torch.full((1, 4, 4), -10.0)
    full_adapter_logits[0, 0, 0] = 9.0
    full_adapter_logits[0, 1, 1] = 8.0
    rank_logits = torch.full((1, 4, 4), float('-inf'))
    rank_logits[0, 0, 0] = 5.0
    rank_logits[0, 1, 1] = 4.0
    rank_logits[0, 2, 2] = 3.0
    rank_logits[0, 3, 3] = 2.0

    audit = prediction_closure_diagnostics(
        tii_ranking=torch.tensor([[0, 1, 2, 3]]),
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=torch.tensor([[5.0, 4.0, 3.0, 2.0]]),
        winner_tasks=torch.tensor([0]),
        full_predictions=torch.tensor([0]),
        class_mask=[[0], [1], [2], [3]],
        initial_count=2,
        top_classes=1,
        tii_logits=torch.tensor([[1.0, 0.5, 4.0, 3.0]]),
        tii_tail_completion=True,
    )

    output = audit['prediction_closure_output_logits']
    assert output.argmax(dim=1).tolist() == [0]
    assert torch.isfinite(output).all()
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(1), atol=1e-6)
    assert audit['prediction_closure_lora_counts'].tolist() == [2.0]
    assert audit['prediction_closure_forward_calls'].tolist() == [1.0]


def test_prediction_beam_closure_follows_only_current_leaders():
    tii_ranking = torch.tensor([[0, 1, 2, 3, 4]])
    full_adapter_logits = torch.full((1, 5, 5), -10.0)
    full_adapter_logits[0, 0, 2] = 9.0
    full_adapter_logits[0, 1, 1] = 9.0
    full_adapter_logits[0, 2, 4] = 9.0
    full_adapter_logits[0, 3, 3] = 9.0
    full_adapter_logits[0, 4, 4] = 9.0
    rank_logits = torch.full((1, 5, 5), float('-inf'))
    rank_logits[0, 0, 0] = 2.0
    rank_logits[0, 1, 1] = 1.0
    rank_logits[0, 2, 2] = 4.0
    rank_logits[0, 3, 3] = 3.0
    rank_logits[0, 4, 4] = 5.0

    audit = prediction_beam_closure_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=torch.tensor([[2.0, 1.0, 4.0, 3.0, 5.0]]),
        winner_tasks=torch.tensor([4]),
        full_predictions=torch.tensor([4]),
        tii_logits=torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]]),
        class_mask=[[0], [1], [2], [3], [4]],
        initial_count=2,
        top_classes=1,
        beam_width=2,
    )

    output = audit['prediction_beam_closure_output_logits']
    assert audit['prediction_beam_closure_winner_recall'].tolist() == [True]
    assert audit['prediction_beam_closure_exact_agreement'].tolist() == [True]
    assert audit['prediction_beam_closure_lora_counts'].tolist() == [4.0]
    assert audit['prediction_beam_closure_forward_calls'].tolist() == [3.0]
    assert audit['prediction_beam_closure_full_scan'].tolist() == [False]
    assert audit['prediction_beam_closure_output_tasks'].tolist() == [4]
    assert output.argmax(dim=1).tolist() == [4]
    assert torch.isfinite(output).all()
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(1), atol=1e-6)


def test_prediction_budget_closure_keeps_strongest_frontier_under_cap():
    tii_ranking = torch.tensor([[0, 1, 2, 3, 4, 5]])
    full_adapter_logits = torch.full((1, 6, 6), -10.0)
    full_adapter_logits[0, 0, 2] = 8.0
    full_adapter_logits[0, 1, 3] = 7.0
    full_adapter_logits[0, 2, 4] = 8.0
    full_adapter_logits[0, 3, 5] = 9.0
    full_adapter_logits[0, 4, 4] = 9.0
    full_adapter_logits[0, 5, 5] = 10.0
    rank_logits = torch.full((1, 6, 6), float('-inf'))
    for rank, evidence in enumerate((2.0, 1.0, 4.0, 3.0, 5.0, 6.0)):
        rank_logits[0, rank, rank] = evidence

    audit = prediction_budget_closure_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=torch.tensor([[2.0, 1.0, 4.0, 3.0, 5.0, 6.0]]),
        winner_tasks=torch.tensor([5]),
        full_predictions=torch.tensor([5]),
        tii_logits=torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
        class_mask=[[0], [1], [2], [3], [4], [5]],
        initial_count=2,
        top_classes=1,
        max_candidates=5,
    )

    output = audit['prediction_budget_closure_output_logits']
    assert audit['prediction_budget_closure_winner_recall'].tolist() == [True]
    assert audit['prediction_budget_closure_exact_agreement'].tolist() == [True]
    assert audit['prediction_budget_closure_lora_counts'].tolist() == [5.0]
    assert audit['prediction_budget_closure_forward_calls'].tolist() == [3.0]
    assert audit['prediction_budget_closure_budget_hit'].tolist() == [True]
    assert audit['prediction_budget_closure_output_tasks'].tolist() == [5]
    assert output.argmax(dim=1).tolist() == [5]
    assert torch.isfinite(output).all()
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(1), atol=1e-6)

def test_prediction_majority_closure_anchors_only_certified_rows():
    tii_ranking = torch.tensor([[0, 1, 2], [0, 1, 2]])
    full_adapter_logits = torch.full((2, 3, 3), -10.0)
    full_adapter_logits[:, 0, 2] = 9.0
    full_adapter_logits[:, 1, 1] = 8.0
    full_adapter_logits[:, 2, 2] = 10.0
    rank_logits = torch.full((2, 3, 3), float('-inf'))
    rank_logits[:, 0, 0] = 4.0
    rank_logits[:, 1, 1] = 2.0
    rank_logits[:, 2, 2] = 5.0

    audit = prediction_majority_budget_closure_diagnostics(
        tii_ranking=tii_ranking,
        full_adapter_logits=full_adapter_logits,
        rank_logits=rank_logits,
        task_evidence=torch.tensor([[4.0, 2.0, 5.0],
                                    [4.0, 2.0, 5.0]]),
        winner_tasks=torch.tensor([2, 2]),
        full_predictions=torch.tensor([2, 2]),
        tii_logits=torch.tensor([[4.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0]]),
        class_mask=[[0], [1], [2]],
        initial_count=2,
        top_classes=1,
        max_candidates=3,
    )

    output = audit['prediction_majority_closure_output_logits']
    assert audit['prediction_majority_closure_majority_rate'].tolist() == [
        True, False]
    assert audit['prediction_majority_closure_lora_counts'].tolist() == [
        1.0, 3.0]
    assert audit['prediction_majority_closure_forward_calls'].tolist() == [
        1.0, 2.0]
    assert audit['prediction_majority_closure_output_tasks'].tolist() == [0, 2]
    assert audit['prediction_majority_closure_winner_recall'].tolist() == [
        False, True]
    assert audit['prediction_majority_closure_exact_agreement'].tolist() == [
        False, True]
    assert output.argmax(dim=1).tolist() == [0, 2]
    assert torch.isfinite(output).all()
    assert torch.allclose(
        output.exp().sum(dim=1), torch.ones(2), atol=1e-6)

def test_prediction_proposal_is_allowed_by_strict_exemplar_free_protocol():
    validate_exemplar_free_protocol(SimpleNamespace(
        strict_exemplar_free=True,
        progressive_oracle_audit=True,
        progressive_prediction_proposal_audit=True,
        progressive_prediction_closure_audit=True,
        progressive_prediction_closure_tii_tail_audit=True,
        progressive_prediction_beam_closure_audit=True,
        progressive_prediction_budget_closure_audit=True,
        progressive_prediction_majority_closure_audit=True,
        prediction_closure_rematching=True,
        prediction_proposal_rematching=True,
        prediction_proposal_cross_adapter_audit=True,
        prediction_proposal_tii_completion=True,
        prediction_proposal_task_mass_fusion=True,
        prediction_proposal_conditional_fusion=True,
        prediction_proposal_crm_confidence_fusion=True,
    ))


class ProposalTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, inputs, task_id):
        self.batch_sizes.append(int(inputs.shape[0]))
        logits = torch.full((inputs.shape[0], 10), -10.0)
        for row, selected_task in enumerate(task_id.tolist()):
            if selected_task == 0:
                logits[row, 0] = 5.0
                logits[row, 8] = 9.0
                logits[row, 4] = 8.0
            elif selected_task == 1:
                logits[row, 2] = 4.0
                logits[row, 9] = 7.0
                logits[row, 5] = 6.0
            elif selected_task == 2:
                logits[row, 4] = 3.0
            elif selected_task == 3:
                logits[row, 6] = 2.0
            else:
                logits[row, 8] = 8.0
        return {'logits': logits}


def test_operational_prediction_proposal_runs_only_four_loras_in_two_calls():
    model = ProposalTaskModel()
    tii_logits = torch.tensor([[
        5.0, 4.9, 4.0, 3.9, 3.0,
        2.9, 2.0, 1.9, 1.0, 0.9,
    ]])
    args = SimpleNamespace(
        progressive_logit_temperature=1.0,
        progressive_tii_prior_weight=0.0,
        progressive_excluded_logit_margin=20.0,
        prediction_proposal_initial_count=2,
        prediction_proposal_count=2,
        prediction_proposal_top_classes=2,
    )

    logits, routed, diagnostics = prediction_proposal_adapter_rematching(
        model=model,
        inputs=torch.tensor([[1.0]]),
        tii_logits=tii_logits,
        class_mask=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        seen_task_count=5,
        args=args,
    )

    assert routed.tolist() == [4]
    assert logits.argmax(dim=1).tolist() == [8]
    assert diagnostics['lora_counts'].tolist() == [4.0]
    assert diagnostics['forward_calls'].tolist() == [2.0]
    assert diagnostics['initial_branch_tasks'].tolist() == [0]
    assert diagnostics['initial_branch_logits'].argmax(dim=1).tolist() == [8]
    assert diagnostics['candidate_logits'].shape == (1, 4, 10)
    assert diagnostics['candidate_tasks'].shape == (1, 4)
    assert model.batch_sizes == [2, 2]


class IterativeProposalTaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, inputs, task_id):
        self.batch_sizes.append(inputs.shape[0])
        logits = torch.zeros((inputs.shape[0], 12), device=inputs.device)
        for row, selected_task in enumerate(task_id.tolist()):
            if selected_task == 0:
                logits[row, 4] = 8.0
            elif selected_task == 1:
                logits[row, 6] = 7.0
            elif selected_task == 2:
                logits[row, 8] = 9.0
            elif selected_task == 3:
                logits[row, 10] = 8.5
            elif selected_task == 4:
                logits[row, 8] = 6.0
            else:
                logits[row, 10] = 10.0
        return {'logits': logits}


def test_iterative_prediction_proposal_discovers_second_wave_tasks():
    model = IterativeProposalTaskModel()
    args = SimpleNamespace(
        progressive_logit_temperature=1.0,
        progressive_tii_prior_weight=0.0,
        progressive_excluded_logit_margin=20.0,
        prediction_proposal_initial_count=2,
        prediction_proposal_count=4,
        prediction_proposal_top_classes=1,
        prediction_proposal_iterative=True,
        prediction_proposal_first_wave_count=2,
    )

    logits, routed, diagnostics = prediction_proposal_adapter_rematching(
        model=model,
        inputs=torch.tensor([[1.0]]),
        tii_logits=torch.arange(12, 0, -1).float().unsqueeze(0),
        class_mask=[
            [0, 1], [2, 3], [4, 5],
            [6, 7], [8, 9], [10, 11],
        ],
        seen_task_count=6,
        args=args,
    )

    assert diagnostics['candidate_tasks'].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert diagnostics['lora_counts'].tolist() == [6.0]
    assert diagnostics['forward_calls'].tolist() == [3.0]
    assert model.batch_sizes == [2, 2, 2]
    assert routed.tolist() == [5]
    assert logits.argmax(dim=1).tolist() == [10]


def test_tii_completion_preserves_top1_and_assigns_finite_outside_mass():
    candidate_logits = torch.tensor([[
        5.0, 4.0, 3.0, 2.0, float('-inf'), float('-inf'),
    ]])
    tii_logits = torch.tensor([[1.0, 0.0, 0.5, 0.0, 4.0, 3.0]])
    candidate_tasks = torch.tensor([[0, 1]])

    completed = _complete_with_tii_probability_mass(
        candidate_logits=candidate_logits,
        tii_logits=tii_logits,
        class_mask=[[0, 1], [2, 3], [4, 5]],
        candidate_tasks=candidate_tasks,
        seen_task_count=3,
    )

    assert completed.argmax(dim=1).tolist() == [0]
    assert torch.isfinite(completed).all()
    assert completed[0, 4] > -20.0
    assert torch.allclose(
        completed.exp().sum(dim=1), torch.ones(1), atol=1e-6)


def test_task_mass_fusion_is_normalized_and_adapter_shift_invariant():
    candidate_logits = torch.tensor([[[4.0, 2.0, 9.0, 8.0],
                                      [7.0, 6.0, 3.0, 1.0]]])
    tii_logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    candidate_tasks = torch.tensor([[0, 1]])
    class_mask = [[0, 1], [2, 3]]

    fused, evidence = task_mass_preserving_candidate_fusion(
        candidate_logits, tii_logits, class_mask, candidate_tasks,
        seen_task_count=2)
    shifted = candidate_logits + torch.tensor([[[100.0], [-50.0]]])
    shifted_fused, shifted_evidence = task_mass_preserving_candidate_fusion(
        shifted, tii_logits, class_mask, candidate_tasks,
        seen_task_count=2)

    assert torch.allclose(fused.exp().sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(fused, shifted_fused, atol=1e-6)
    assert torch.allclose(evidence, shifted_evidence, atol=1e-6)


def test_conditional_fusion_is_adapter_shift_invariant():
    candidate_logits = torch.tensor([[[4.0, 2.0, 9.0, 8.0],
                                      [7.0, 6.0, 3.0, 1.0]]])
    tii_prior = torch.tensor([[0.5, -0.5]])
    candidate_tasks = torch.tensor([[0, 1]])
    class_mask = [[0, 1], [2, 3]]

    fused, evidence = conditional_candidate_fusion(
        candidate_logits, tii_prior, class_mask, candidate_tasks,
        prior_weight=0.3)
    shifted = candidate_logits + torch.tensor([[[100.0], [-50.0]]])
    shifted_fused, shifted_evidence = conditional_candidate_fusion(
        shifted, tii_prior, class_mask, candidate_tasks,
        prior_weight=0.3)

    assert torch.allclose(fused, shifted_fused, atol=1e-6)
    assert torch.allclose(evidence, shifted_evidence, atol=1e-6)


def test_crm_confidence_fusion_is_normalized_and_adapter_shift_invariant():
    candidate_logits = torch.tensor([[[4.0, 1.0, 9.0, 8.0],
                                      [7.0, 6.0, 3.0, 2.5]]])
    tii_prior = torch.zeros((1, 2))
    candidate_tasks = torch.tensor([[0, 1]])
    class_mask = [[0, 1], [2, 3]]

    fused, evidence = crm_confidence_candidate_fusion(
        candidate_logits, tii_prior, class_mask, candidate_tasks,
        confidence_temperature=0.1, prior_weight=0.0)
    shifted = candidate_logits + torch.tensor([[[100.0], [-50.0]]])
    shifted_fused, shifted_evidence = crm_confidence_candidate_fusion(
        shifted, tii_prior, class_mask, candidate_tasks,
        confidence_temperature=0.1, prior_weight=0.0)

    assert torch.allclose(
        fused.exp().sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(fused, shifted_fused, atol=1e-6)
    assert torch.allclose(evidence, shifted_evidence, atol=1e-6)
    assert evidence[0, 0] > evidence[0, 1]


def test_crm_confidence_fusion_uses_tii_prior_only_as_a_soft_tie_break():
    candidate_logits = torch.tensor([[[3.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 3.0, 1.0]]])
    candidate_tasks = torch.tensor([[0, 1]])
    class_mask = [[0, 1], [2, 3]]

    fused, evidence = crm_confidence_candidate_fusion(
        candidate_logits,
        tii_prior=torch.tensor([[0.0, 2.0]]),
        class_mask=class_mask,
        candidate_tasks=candidate_tasks,
        confidence_temperature=0.1,
        prior_weight=0.3,
    )

    assert evidence[0, 1] > evidence[0, 0]
    assert fused.argmax(dim=1).tolist() == [2]


def test_initial_branch_selector_requires_confidence_and_margin_dominance():
    initial_logits = torch.tensor([
        [4.0, 0.0],
        [2.0, 1.0],
        [3.0, 0.0],
    ])
    proposal_logits = torch.tensor([
        [0.0, 2.0],
        [0.0, 4.0],
        [2.0, 0.0],
    ])

    selected = initial_branch_confidence_dominance(
        initial_logits, proposal_logits)

    assert selected.tolist() == [True, False, False]


def test_cross_adapter_consensus_uses_plurality_and_probability_tie_break():
    candidate_logits = torch.tensor([
        [
            [5.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 2.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        [
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    ])

    audit = cross_adapter_global_consensus(candidate_logits)

    assert audit['consensus_prediction'].tolist() == [0, 1]
    assert audit['strict_majority'].tolist() == [True, False]
    assert torch.allclose(
        audit['vote_strength'], torch.tensor([0.6, 0.4]))


def test_cross_adapter_borda_rewards_consistent_high_rank_support():
    candidate_logits = torch.tensor([[
        [5.0, 4.0, 0.0],
        [5.0, 4.0, 0.0],
        [0.0, 4.0, 5.0],
        [0.0, 4.0, 5.0],
        [0.0, 5.0, 4.0],
    ]])

    audit = cross_adapter_borda_consensus(candidate_logits, top_k=2)

    assert audit['prediction'].tolist() == [1]
    assert audit['strict_support'].tolist() == [True]
    assert torch.allclose(audit['topk_support'], torch.ones(1))
