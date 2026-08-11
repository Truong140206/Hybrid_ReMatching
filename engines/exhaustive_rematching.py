import torch


def _standardize_task_scores(scores):
    if scores.shape[1] <= 1:
        return torch.zeros_like(scores)
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


@torch.no_grad()
def exhaustive_adapter_rematching(model, inputs, tii_logits, class_mask,
                                  seen_task_count, args):
    """Score each task's classes with its own LoRA, then merge all class logits."""
    device = inputs.device
    temperature = max(
        1e-6, float(getattr(args, 'exhaustive_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'exhaustive_tii_prior_weight', 0.1)))

    tii_task_scores = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        task_logits = tii_logits.index_select(1, class_index)
        tii_task_scores.append(task_logits.max(dim=1).values)
    tii_task_prior = _standardize_task_scores(
        torch.stack(tii_task_scores, dim=1))

    merged_logits = torch.full_like(tii_logits, float('-inf'))
    task_evidence = []
    for task_index in range(seen_task_count):
        task_ids = torch.full(
            (inputs.shape[0],), task_index, dtype=torch.long, device=device)
        adapter_logits = model(inputs, task_id=task_ids)['logits']
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        local_logits = adapter_logits.index_select(1, class_index) / temperature
        local_logits = (
            local_logits
            + prior_weight * tii_task_prior[:, task_index].unsqueeze(1)
        )
        merged_logits[:, class_index] = local_logits
        task_evidence.append(local_logits.max(dim=1).values)

    routed_tasks = torch.stack(task_evidence, dim=1).argmax(dim=1)
    return merged_logits, routed_tasks
