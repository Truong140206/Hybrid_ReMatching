import torch
from torch.nn import functional as F

from engines.progressive_rematching import _tii_task_prior


@torch.no_grad()
def soft_mixture_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Mix the TII top-k LoRA residuals in one model forward pass."""
    device = inputs.device
    batch_size = inputs.shape[0]
    top_k = min(
        int(seen_task_count),
        max(1, int(getattr(args, 'soft_mixture_top_k', 4))),
    )
    task_temperature = max(
        1e-6, float(getattr(args, 'soft_mixture_task_temperature', 1.0)))
    logit_temperature = max(
        1e-6, float(getattr(args, 'soft_mixture_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'soft_mixture_tii_prior_weight', 0.3)))

    tii_prior = _tii_task_prior(tii_logits, class_mask, seen_task_count)
    top_scores, candidate_tasks = torch.topk(
        tii_prior, k=top_k, dim=1, largest=True, sorted=True)
    mixture_weights = F.softmax(top_scores / task_temperature, dim=1)

    output = model(
        inputs,
        task_id=candidate_tasks[:, 0],
        ensemble_id=candidate_tasks,
        ensemble_weights=mixture_weights,
    )
    mixed_logits = output['logits'] / logit_temperature

    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        local_logits = mixed_logits.index_select(1, class_index)
        calibrated_logits = (
            local_logits
            + prior_weight * tii_prior[:, task_index].unsqueeze(1)
        )
        merged_logits[:, class_index] = calibrated_logits
        task_evidence.append(calibrated_logits.max(dim=1).values)

    routed_tasks = torch.stack(task_evidence, dim=1).argmax(dim=1)
    diagnostics = {
        'candidate_tasks': candidate_tasks,
        'mixture_weights': mixture_weights,
        'lora_counts': torch.full(
            (batch_size,), float(top_k), dtype=torch.float32, device=device),
        'forward_calls': torch.ones(
            batch_size, dtype=torch.float32, device=device),
    }
    return merged_logits, routed_tasks, diagnostics


@torch.no_grad()
def soft_mixture_hard_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args):
    """Route with a soft LoRA mixture, then classify with one hard LoRA."""
    _, routed_tasks, diagnostics = soft_mixture_adapter_rematching(
        model, inputs, tii_logits, class_mask, seen_task_count, args)

    hard_output = model(inputs, task_id=routed_tasks)
    hard_logits = hard_output['logits'] / max(
        1e-6, float(getattr(args, 'soft_mixture_logit_temperature', 1.0)))
    merged_logits = torch.full_like(tii_logits, float('-inf'))
    seen_classes = []
    for task_index in range(seen_task_count):
        seen_classes.extend(class_mask[task_index])
    seen_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=inputs.device)
    merged_logits[:, seen_index] = hard_logits.index_select(1, seen_index)

    diagnostics = dict(diagnostics)
    diagnostics['lora_counts'] = diagnostics['lora_counts'] + 1.0
    diagnostics['forward_calls'] = diagnostics['forward_calls'] + 1.0
    return merged_logits, routed_tasks, diagnostics
