import torch

from engines.hrm_lora_wtp_and_tap_engine import (
    compute_relation_matrix,
    online_ctird_rank_weight,
    select_ctird_source_tasks,
)


def test_source_selection_aggregates_evidence_per_task():
    class_mask = [[0, 1], [2, 3], [4, 5]]
    logits = torch.tensor([
        [3.5, 3.5, 4.0, -4.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
    ])

    selected = select_ctird_source_tasks(
        logits, class_mask, num_old_tasks=3, top_k=2, temperature=1.0)

    # Sample 0 prefers task 0 by aggregate mass even though task 1 owns
    # the single largest class logit. Sample 1 prefers task 2 then task 1.
    assert selected.tolist() == [[0, 1], [2, 1]]


def test_source_selection_returns_unique_tasks_and_respects_old_task_limit():
    class_mask = [[0, 1], [2, 3], [4, 5]]
    logits = torch.tensor([[0.0, 0.0, 2.0, 2.0, 10.0, 10.0]])

    selected = select_ctird_source_tasks(
        logits, class_mask, num_old_tasks=2, top_k=5, temperature=1.0)

    assert selected.shape == (1, 2)
    assert selected.tolist() == [[1, 0]]


def test_relation_matrix_is_a_finite_row_distribution():
    features = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    relation = compute_relation_matrix(features)

    assert torch.isfinite(relation).all()
    assert torch.all(relation > 0)
    assert torch.allclose(relation.sum(dim=1), torch.ones(3))


def test_online_ctird_rank_weight_supports_sum_and_mean_reductions():
    assert online_ctird_rank_weight(5, 1, reduction='sum') == 5.0
    assert online_ctird_rank_weight(5, 2, reduction='sum') == 2.5
    assert online_ctird_rank_weight(5, 1, reduction='mean') == 1.0
    assert online_ctird_rank_weight(5, 2, reduction='mean') == 0.5
