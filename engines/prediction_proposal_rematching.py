import torch

from engines.progressive_oracle_audit import (
    _finalize_partial_logits,
    prediction_proposal_candidates,
)
from engines.progressive_rematching import _tii_task_prior


def _evaluate_candidate_tasks(model, inputs, candidate_tasks):
    batch_size, candidate_count = candidate_tasks.shape
    expanded_inputs = inputs.unsqueeze(1).expand(
        batch_size, candidate_count, *inputs.shape[1:]
    ).reshape(batch_size * candidate_count, *inputs.shape[1:])
    expanded_task_ids = candidate_tasks.reshape(-1)
    return model(
        expanded_inputs, task_id=expanded_task_ids
    )['logits'].reshape(batch_size, candidate_count, -1)


def initial_branch_confidence_dominance(initial_logits, proposal_logits):
    """Select initial output only when it Pareto-dominates proposal confidence."""
    initial_probabilities = torch.softmax(initial_logits, dim=1)
    proposal_probabilities = torch.softmax(proposal_logits, dim=1)
    initial_top2 = torch.topk(initial_probabilities, k=2, dim=1).values
    proposal_top2 = torch.topk(proposal_probabilities, k=2, dim=1).values
    initial_prediction = initial_logits.argmax(dim=1)
    proposal_prediction = proposal_logits.argmax(dim=1)
    return (
        initial_prediction.ne(proposal_prediction)
        & initial_top2[:, 0].gt(proposal_top2[:, 0])
        & (initial_top2[:, 0] - initial_top2[:, 1]).gt(
            proposal_top2[:, 0] - proposal_top2[:, 1])
    )


def cross_adapter_global_consensus(candidate_logits):
    """Plurality vote over full adapter predictions with probability tie-break."""
    batch_size, candidate_count, class_count = candidate_logits.shape
    probabilities = torch.softmax(candidate_logits, dim=2)
    predictions = candidate_logits.argmax(dim=2)
    vote_counts = torch.zeros(
        (batch_size, class_count), dtype=torch.long,
        device=candidate_logits.device)
    vote_counts.scatter_add_(
        1, predictions,
        torch.ones_like(predictions, dtype=torch.long))
    max_votes = vote_counts.max(dim=1).values
    tied_classes = vote_counts.eq(max_votes.unsqueeze(1))
    probability_support = probabilities.sum(dim=1)
    consensus_prediction = probability_support.masked_fill(
        ~tied_classes, float('-inf')).argmax(dim=1)
    strict_majority = max_votes.mul(2).gt(candidate_count)
    vote_strength = max_votes.float().div(float(candidate_count))
    return {
        'adapter_predictions': predictions,
        'consensus_prediction': consensus_prediction,
        'strict_majority': strict_majority,
        'vote_strength': vote_strength,
    }


def cross_adapter_borda_consensus(candidate_logits, top_k=5):
    """Calibration-free Borda aggregation over full class rankings."""
    _, candidate_count, class_count = candidate_logits.shape
    ordering = torch.argsort(candidate_logits, dim=2, descending=True)
    ranks = torch.argsort(ordering, dim=2)
    rank_sum = ranks.sum(dim=1)
    best_rank_sum = rank_sum.min(dim=1).values
    tied_classes = rank_sum.eq(best_rank_sum.unsqueeze(1))
    probability_support = torch.softmax(candidate_logits, dim=2).sum(dim=1)
    prediction = probability_support.masked_fill(
        ~tied_classes, float('-inf')).argmax(dim=1)
    selected_ranks = ranks.gather(
        2,
        prediction.view(-1, 1, 1).expand(-1, candidate_count, 1),
    ).squeeze(2)
    top_k = min(max(1, int(top_k)), class_count)
    support_count = selected_ranks.lt(top_k).sum(dim=1)
    strict_support = support_count.mul(2).gt(candidate_count)
    return {
        'prediction': prediction,
        'topk_support': support_count.float().div(float(candidate_count)),
        'strict_support': strict_support,
    }


def _complete_with_tii_probability_mass(
        candidate_logits, tii_logits, class_mask, candidate_tasks,
        seen_task_count):
    """Give excluded seen classes TII mass without changing candidate top-1."""
    batch_size, class_count = candidate_logits.shape
    device = candidate_logits.device
    included_mask = torch.zeros(
        (batch_size, class_count), dtype=torch.bool, device=device)
    for candidate_slot in range(candidate_tasks.shape[1]):
        active_tasks = candidate_tasks[:, candidate_slot]
        for task_index in active_tasks.unique().tolist():
            rows = torch.nonzero(active_tasks == task_index).flatten()
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            included_mask[
                rows.unsqueeze(1), class_index.unsqueeze(0)
            ] = True

    seen_class_mask = torch.zeros(
        class_count, dtype=torch.bool, device=device)
    for task_classes in class_mask[:seen_task_count]:
        seen_class_mask[torch.as_tensor(
            task_classes, dtype=torch.long, device=device)] = True
    outside_mask = seen_class_mask.unsqueeze(0) & ~included_mask

    candidate_probabilities = torch.softmax(candidate_logits, dim=1)
    masked_tii_logits = tii_logits.masked_fill(
        ~seen_class_mask.unsqueeze(0), float('-inf'))
    tii_probabilities = torch.softmax(masked_tii_logits, dim=1)
    outside_probabilities = tii_probabilities.masked_fill(~outside_mask, 0.0)
    outside_mass = outside_probabilities.sum(dim=1, keepdim=True)
    outside_conditional = outside_probabilities / outside_mass.clamp_min(1e-12)

    candidate_peak = candidate_probabilities.max(dim=1, keepdim=True).values
    outside_peak = outside_conditional.max(dim=1, keepdim=True).values
    top1_safe_mass = 0.99 * candidate_peak / (
        candidate_peak + outside_peak).clamp_min(1e-12)
    completion_mass = torch.minimum(outside_mass, top1_safe_mass)
    completed_probabilities = (
        (1.0 - completion_mass) * candidate_probabilities
        + completion_mass * outside_conditional
    )
    return completed_probabilities.clamp_min(1e-12).log()


@torch.no_grad()
def prediction_proposal_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Route through TII top tasks plus post-LoRA prediction proposals."""
    device = inputs.device
    batch_size = inputs.shape[0]
    temperature = max(
        1e-6, float(getattr(args, 'progressive_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'progressive_tii_prior_weight', 0.3)))
    excluded_margin = max(
        1.0, float(getattr(args, 'progressive_excluded_logit_margin', 20.0)))
    initial_count = min(
        max(1, int(getattr(args, 'prediction_proposal_initial_count', 2))),
        seen_task_count)
    proposal_count = min(
        max(0, int(getattr(args, 'prediction_proposal_count', 2))),
        seen_task_count - initial_count)
    top_classes = max(
        1, int(getattr(args, 'prediction_proposal_top_classes', 5)))

    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    tii_ranking = torch.argsort(tii_prior, dim=1, descending=True)
    initial_tasks = tii_ranking[:, :initial_count]
    initial_logits = _evaluate_candidate_tasks(
        model, inputs, initial_tasks)
    candidate_tasks, _ = prediction_proposal_candidates(
        tii_ranking,
        initial_logits,
        class_mask,
        initial_count=initial_count,
        proposal_count=proposal_count,
        top_classes=top_classes,
    )

    candidate_logits = initial_logits
    forward_calls = 1
    if proposal_count > 0:
        proposal_tasks = candidate_tasks[:, initial_count:]
        proposal_logits = _evaluate_candidate_tasks(
            model, inputs, proposal_tasks)
        candidate_logits = torch.cat(
            [candidate_logits, proposal_logits], dim=1)
        forward_calls += 1

    candidate_count = candidate_tasks.shape[1]
    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = torch.full(
        (batch_size, candidate_count), float('-inf'),
        dtype=tii_logits.dtype, device=device)
    for candidate_slot in range(candidate_count):
        active_tasks = candidate_tasks[:, candidate_slot]
        adapter_logits = candidate_logits[:, candidate_slot]
        for task_index in active_tasks.unique().tolist():
            rows = torch.nonzero(active_tasks == task_index).flatten()
            class_index = torch.as_tensor(
                class_mask[task_index], dtype=torch.long, device=device)
            local_logits = adapter_logits.index_select(
                0, rows).index_select(1, class_index) / temperature
            calibrated_logits = local_logits + prior_weight * tii_prior[
                rows, task_index].unsqueeze(1)
            merged_logits[
                rows.unsqueeze(1), class_index.unsqueeze(0)
            ] = calibrated_logits
            task_evidence[rows, candidate_slot] = calibrated_logits.max(
                dim=1).values

    selected_slot = task_evidence.argmax(dim=1)
    routed_tasks = candidate_tasks.gather(
        1, selected_slot.unsqueeze(1)).squeeze(1)
    if bool(getattr(args, 'prediction_proposal_tii_completion', False)):
        merged_logits = _complete_with_tii_probability_mass(
            merged_logits, tii_logits, class_mask, candidate_tasks,
            seen_task_count)
    else:
        merged_logits = _finalize_partial_logits(
            merged_logits, excluded_margin)
    diagnostics = {
        'lora_counts': torch.full(
            (batch_size,), float(candidate_count),
            dtype=torch.float32, device=device),
        'forward_calls': torch.full(
            (batch_size,), float(forward_calls),
            dtype=torch.float32, device=device),
        # The first evaluated task is exactly the initial TII top-1
        # route, so exposing it for audit adds neither a LoRA nor a forward.
        'initial_branch_logits': initial_logits[:, 0],
        'initial_branch_tasks': initial_tasks[:, 0],
        # Full logits already exist for every evaluated adapter. Keeping them
        # transiently for audit adds no model execution or persistent memory.
        'candidate_logits': candidate_logits,
        'candidate_tasks': candidate_tasks,
    }
    return merged_logits, routed_tasks, diagnostics
