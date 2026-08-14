from types import SimpleNamespace

import torch
from torch import nn

from engines.exhaustive_rematching import exhaustive_adapter_rematching
from engines.vectorized_exhaustive_rematching import (
    vectorized_exhaustive_adapter_rematching,
)


class TinyTaskModel(nn.Module):
    def forward(self, inputs, task_id):
        task_id = task_id.to(dtype=inputs.dtype).unsqueeze(1)
        base = inputs[:, :1]
        logits = torch.cat([
            base + task_id,
            base - task_id,
            -base + 0.5 * task_id,
            -base - 0.5 * task_id,
        ], dim=1)
        return {'logits': logits, 'pre_logits': inputs}


def _args(chunk_size):
    return SimpleNamespace(
        exhaustive_logit_temperature=1.0,
        exhaustive_tii_prior_weight=0.3,
        exhaustive_max_calibration_weight=0.5,
        exhaustive_tii_class_weight=0.0,
        exhaustive_tii_class_temperature=1.0,
        exhaustive_local_prototype_weight=0.0,
        exhaustive_local_prototype_temperature=0.07,
        vectorized_exhaustive_task_chunk_size=chunk_size,
    )


def test_vectorized_exhaustive_is_exact_for_multiple_chunk_sizes():
    model = TinyTaskModel().eval()
    inputs = torch.tensor([[0.2], [-0.4], [1.0]])
    tii_logits = torch.tensor([
        [0.5, 0.2, -0.1, -0.3],
        [0.1, 0.4, -0.2, 0.0],
        [0.9, 0.7, 0.1, -0.4],
    ])
    class_mask = [[0, 1], [2, 3]]
    reference_logits, reference_tasks = exhaustive_adapter_rematching(
        model, inputs, tii_logits, class_mask, 2, _args(1))

    for chunk_size in (1, 2):
        logits, tasks, diagnostics = (
            vectorized_exhaustive_adapter_rematching(
                model, inputs, tii_logits, class_mask, 2,
                _args(chunk_size)))
        assert torch.equal(tasks, reference_tasks)
        assert torch.allclose(logits, reference_logits, atol=1e-7, rtol=0.0)
        assert diagnostics['lora_counts'].tolist() == [2.0, 2.0, 2.0]
        expected_calls = 2.0 if chunk_size == 1 else 1.0
        assert diagnostics['forward_calls'].tolist() == [
            expected_calls, expected_calls, expected_calls]
