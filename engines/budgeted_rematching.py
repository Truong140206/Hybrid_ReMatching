import torch


def _standardize_task_scores(scores):
    if scores.shape[1] <= 1:
        return torch.zeros_like(scores)
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (scores - mean) / std


def _select_fallback_samples(task_scores, fallback_fraction):
    batch_size, task_count = task_scores.shape
    fallback_mask = torch.zeros(
        batch_size, dtype=torch.bool, device=task_scores.device)
    if task_count <= 1 or fallback_fraction <= 0.0:
        return fallback_mask

    top_values = torch.topk(task_scores, k=2, dim=1, largest=True).values
    confidence_margin = top_values[:, 0] - top_values[:, 1]
    fallback_count = int(round(batch_size * fallback_fraction))
    fallback_count = min(batch_size, max(1, fallback_count))
    fallback_index = torch.topk(
        confidence_margin, k=fallback_count, largest=False).indices
    fallback_mask[fallback_index] = True
    return fallback_mask


@torch.no_grad()
def budgeted_adapter_rematching(model, inputs, tii_logits, class_mask,
                                seen_task_count, args):
    """Use top-1 routing normally and exhaustive rematching on uncertain samples."""
    device = inputs.device
    temperature = max(
        1e-6, float(getattr(args, 'budgeted_logit_temperature', 1.0)))
    prior_weight = max(
        0.0, float(getattr(args, 'budgeted_tii_prior_weight', 0.3)))
    fallback_fraction = min(
        1.0, max(0.0, float(
            getattr(args, 'budgeted_fallback_fraction', 0.2))))

    tii_task_scores = []
    seen_classes = []
    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        seen_classes.extend(int(value) for value in class_mask[task_index])
        task_logits = tii_logits.index_select(1, class_index)
        tii_task_scores.append(task_logits.max(dim=1).values)
    tii_task_prior = _standardize_task_scores(
        torch.stack(tii_task_scores, dim=1))
    top1_tasks = tii_task_prior.argmax(dim=1)
    fallback_mask = _select_fallback_samples(
        tii_task_prior, fallback_fraction)

    top1_output = model(inputs, task_id=top1_tasks)['logits']
    logits = torch.full_like(top1_output, float('-inf'))
    seen_index = torch.as_tensor(
        seen_classes, dtype=torch.long, device=device)
    logits[:, seen_index] = top1_output.index_select(1, seen_index)
    routed_tasks = top1_tasks.clone()
    lora_counts = torch.ones(
        inputs.shape[0], dtype=torch.float32, device=device)

    fallback_index = torch.nonzero(fallback_mask).flatten()
    if fallback_index.numel() == 0:
        return logits, routed_tasks, fallback_mask, lora_counts

    fallback_inputs = inputs.index_select(0, fallback_index)
    fallback_top1 = top1_tasks.index_select(0, fallback_index)
    fallback_top1_logits = top1_output.index_select(0, fallback_index)
    merged_logits = torch.full(
        (fallback_index.numel(), tii_logits.shape[1]),
        float('-inf'), dtype=tii_logits.dtype, device=device)
    task_evidence = []

    for task_index in range(seen_task_count):
        class_index = torch.as_tensor(
            class_mask[task_index], dtype=torch.long, device=device)
        local_logits = fallback_top1_logits.index_select(1, class_index).clone()
        rerun_rows = torch.nonzero(fallback_top1 != task_index).flatten()
        if rerun_rows.numel() > 0:
            task_ids = torch.full(
                (rerun_rows.numel(),), task_index,
                dtype=torch.long, device=device)
            adapter_logits = model(
                fallback_inputs.index_select(0, rerun_rows),
                task_id=task_ids)['logits']
            local_logits[rerun_rows] = adapter_logits.index_select(
                1, class_index)

        local_logits = local_logits / temperature
        local_logits = local_logits + prior_weight * tii_task_prior[
            fallback_index, task_index].unsqueeze(1)
        merged_logits[:, class_index] = local_logits
        task_evidence.append(local_logits.max(dim=1).values)

    logits[fallback_index] = merged_logits
    routed_tasks[fallback_index] = torch.stack(
        task_evidence, dim=1).argmax(dim=1)
    lora_counts[fallback_index] = float(seen_task_count)
    return logits, routed_tasks, fallback_mask, lora_counts
