import torch

from engines.arrow_lora_audit import (
    arrow_candidate_diagnostics,
    arrow_task_scores,
)
from engines.progressive_rematching import _tii_task_prior


def _finalize_partial_logits(partial_logits, excluded_margin):
    finite_mask = torch.isfinite(partial_logits)
    row_min = partial_logits.masked_fill(~finite_mask, float('inf')).min(
        dim=1, keepdim=True).values
    return torch.where(
        finite_mask, partial_logits, row_min - excluded_margin)


@torch.no_grad()
def progressive_oracle_audit(model, inputs, tii_logits, class_mask,
                             seen_task_count, args):
    """Run exhaustive once and measure the best possible 2->4->all cascade."""
    device = inputs.device
    batch_size = inputs.shape[0]
    temperature = max(
        1e-6, float(getattr(args, 'progressive_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'progressive_tii_prior_weight', 0.3)))
    excluded_margin = max(
        1.0, float(getattr(args, 'progressive_excluded_logit_margin', 20.0)))

    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    _, candidate_tasks = torch.sort(tii_prior, dim=1, descending=True)

    rank_logits = torch.full(
        (batch_size, seen_task_count, tii_logits.shape[1]),
        float('-inf'), dtype=tii_logits.dtype, device=device)
    task_evidence = torch.full(
        (batch_size, seen_task_count), float('-inf'),
        dtype=tii_logits.dtype, device=device)

    for rank in range(seen_task_count):
        active_tasks = candidate_tasks[:, rank]
        adapter_logits = model(inputs, task_id=active_tasks)['logits']
        for task_index in active_tasks.unique().tolist():
            rows = torch.nonzero(active_tasks == task_index).flatten()
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            local_logits = adapter_logits.index_select(
                0, rows).index_select(1, class_index) / temperature
            calibrated_logits = local_logits + prior_weight * tii_prior[
                rows, task_index].unsqueeze(1)
            rank_logits[
                rows.unsqueeze(1), rank, class_index.unsqueeze(0)
            ] = calibrated_logits
            task_evidence[rows, rank] = calibrated_logits.max(dim=1).values

    full_logits = rank_logits.max(dim=1).values
    selected_rank = task_evidence.argmax(dim=1)
    routed_tasks = candidate_tasks.gather(
        1, selected_rank.unsqueeze(1)).squeeze(1)
    full_predictions = full_logits.argmax(dim=1)

    boundaries = sorted(set([
        min(2, seen_task_count),
        min(4, seen_task_count),
        seen_task_count,
    ]))
    partial_results = {}
    oracle_counts = torch.full(
        (batch_size,), float(seen_task_count),
        dtype=torch.float32, device=device)
    unresolved = torch.ones(batch_size, dtype=torch.bool, device=device)

    for boundary in boundaries:
        partial_logits = rank_logits[:, :boundary].max(dim=1).values
        partial_logits = _finalize_partial_logits(
            partial_logits, excluded_margin)
        partial_rank = task_evidence[:, :boundary].argmax(dim=1)
        partial_tasks = candidate_tasks[:, :boundary].gather(
            1, partial_rank.unsqueeze(1)).squeeze(1)
        prediction_agreement = partial_logits.argmax(dim=1).eq(full_predictions)
        route_agreement = partial_tasks.eq(routed_tasks)
        exact_agreement = prediction_agreement & route_agreement
        partial_results[boundary] = {
            'prediction_agreement': prediction_agreement,
            'route_agreement': route_agreement,
            'exact_agreement': exact_agreement,
        }
        newly_resolved = unresolved & exact_agreement
        oracle_counts[newly_resolved] = float(boundary)
        unresolved[newly_resolved] = False

    winner_rank = selected_rank + 1
    boundary2 = min(2, seen_task_count)
    boundary4 = min(4, seen_task_count)
    diagnostics = {
        'winner_recall_2': winner_rank.le(boundary2),
        'winner_recall_4': winner_rank.le(boundary4),
        'exact_agreement_2': partial_results[boundary2]['exact_agreement'],
        'exact_agreement_4': partial_results[boundary4]['exact_agreement'],
        'prediction_agreement_2': partial_results[boundary2][
            'prediction_agreement'],
        'prediction_agreement_4': partial_results[boundary4][
            'prediction_agreement'],
        'oracle_lora_counts': oracle_counts,
        'actual_lora_counts': torch.full_like(
            oracle_counts, float(seen_task_count)),
    }
    if bool(getattr(args, 'progressive_arrow_audit', False)):
        arrow_scores = arrow_task_scores(model, inputs, seen_task_count)
        arrow_ranking = torch.argsort(arrow_scores, dim=1, descending=True)
        diagnostics.update(arrow_candidate_diagnostics(
            candidate_tasks, arrow_ranking, routed_tasks))
    return full_logits, routed_tasks, diagnostics
