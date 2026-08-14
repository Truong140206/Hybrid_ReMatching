from types import SimpleNamespace

import torch
from torch import nn

from engines.soft_mixture_rematching import (
    _scale_invariant_top1_margin,
    soft_mixture_adapter_rematching,
    soft_hard_confidence_selector_rematching,
    soft_mixture_hard_adapter_rematching,
)
from vits.hrm_lora_vision_transformer import Attention


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_tasks = None
        self.last_weights = None
        self.hard_tasks = None

    def forward(
            self, inputs, task_id, ensemble_id=None, ensemble_weights=None):
        self.call_count += 1
        if ensemble_id is None:
            self.hard_tasks = task_id.detach().clone()
            task_signal = task_id.to(inputs.dtype).unsqueeze(1)
            logits = torch.cat([inputs + task_signal] * 4, dim=1)
            return {'logits': logits}
        self.last_tasks = ensemble_id.detach().clone()
        self.last_weights = ensemble_weights.detach().clone()
        task_signal = (ensemble_id.to(inputs.dtype) * ensemble_weights).sum(
            dim=1, keepdim=True)
        logits = torch.cat([
            inputs[:, :1] + task_signal,
            inputs[:, :1] - task_signal,
            -inputs[:, :1] + task_signal,
            -inputs[:, :1] - task_signal,
        ], dim=1)
        return {'logits': logits}


class ConstantValueLora:
    def __call__(self, x, task_id, **kwargs):
        batch_size, token_count, dim = x.shape
        value = (task_id.to(x.dtype) + 1.0).view(batch_size, 1, 1)
        qkv = torch.zeros(
            batch_size, token_count, dim * 3,
            dtype=x.dtype, device=x.device)
        qkv[:, :, 2 * dim:] = value
        return {'lora_value': qkv}


class SelectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(
            self, inputs, task_id, ensemble_id=None, ensemble_weights=None):
        self.call_count += 1
        if ensemble_id is not None:
            logits = torch.tensor(
                [[3.0, 1.0, 0.0], [1.0, 0.9, 0.0]], device=inputs.device)
        else:
            logits = torch.tensor(
                [[1.0, 0.9, 0.0], [3.0, 1.0, 0.0]], device=inputs.device)
        return {'logits': logits.to(dtype=inputs.dtype)}


def _args():
    return SimpleNamespace(
        soft_mixture_top_k=2,
        soft_mixture_task_temperature=1.0,
        soft_mixture_logit_temperature=1.0,
        soft_mixture_tii_prior_weight=0.3,
    )


def test_soft_mixture_uses_one_forward_and_normalized_topk_weights():
    model = RecordingModel().eval()
    inputs = torch.tensor([[0.2], [-0.4]])
    tii_logits = torch.tensor([
        [0.2, 0.1, 1.0, 0.8],
        [1.1, 0.7, 0.0, -0.2],
    ])
    class_mask = [[0, 1], [2, 3]]

    logits, routed_tasks, diagnostics = soft_mixture_adapter_rematching(
        model, inputs, tii_logits, class_mask, 2, _args())

    assert model.call_count == 1
    assert model.last_tasks.tolist() == [[1, 0], [0, 1]]
    assert torch.allclose(model.last_weights.sum(dim=1), torch.ones(2))
    assert torch.isfinite(logits).all()
    assert routed_tasks.shape == (2,)
    assert diagnostics['lora_counts'].tolist() == [2.0, 2.0]
    assert diagnostics['forward_calls'].tolist() == [1.0, 1.0]


def test_attention_applies_per_sample_convex_lora_weights():
    attention = Attention(dim=2, num_heads=1, qkv_bias=False).eval()
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.proj.weight.copy_(torch.eye(2))
        attention.proj.bias.zero_()

    inputs = torch.zeros(1, 1, 2)
    output = attention(
        inputs,
        ensemble_id=torch.tensor([[0, 1]]),
        ensemble_weights=torch.tensor([[0.25, 0.75]]),
        lora=ConstantValueLora(),
        depth_id=0,
        train=False,
        old=False,
    )

    assert torch.allclose(output, torch.full_like(output, 1.75), atol=1e-6)


def test_soft_mixture_hard_classification_uses_selected_lora_second():
    model = RecordingModel().eval()
    inputs = torch.tensor([[0.2], [-0.4]])
    tii_logits = torch.tensor([
        [0.2, 0.1, 1.0, 0.8, 100.0],
        [1.1, 0.7, 0.0, -0.2, 100.0],
    ])
    class_mask = [[0, 1], [2, 3]]

    logits, routed_tasks, diagnostics = (
        soft_mixture_hard_adapter_rematching(
            model, inputs, tii_logits, class_mask, 2, _args()))

    assert model.call_count == 2
    assert torch.equal(model.hard_tasks, routed_tasks)
    assert diagnostics['lora_counts'].tolist() == [3.0, 3.0]
    assert diagnostics['forward_calls'].tolist() == [2.0, 2.0]
    assert torch.isfinite(logits[:, :4]).all()
    assert torch.isneginf(logits[:, 4]).all()
    expected = inputs + routed_tasks.to(inputs.dtype).unsqueeze(1)
    expected = expected.expand(-1, 4)
    assert torch.allclose(logits[:, :4], expected)


def test_scale_invariant_margin_ignores_shift_and_positive_scale():
    logits = torch.tensor([[3.0, 1.0, -2.0], [8.0, 4.0, -6.0]])
    class_index = torch.arange(3)
    margin = _scale_invariant_top1_margin(logits, class_index)
    transformed = _scale_invariant_top1_margin(
        logits * 7.0 + 13.0, class_index)

    assert torch.allclose(margin, transformed, atol=1e-6)


def test_confidence_selector_uses_larger_normalized_margin_without_labels():
    model = SelectorModel().eval()
    inputs = torch.zeros(2, 1)
    tii_logits = torch.zeros(2, 3)
    class_mask = [[0, 1, 2]]

    logits, routed_tasks, diagnostics = (
        soft_hard_confidence_selector_rematching(
            model, inputs, tii_logits, class_mask, 1, _args()))

    assert model.call_count == 2
    assert routed_tasks.tolist() == [0, 0]
    assert diagnostics['select_hard'].tolist() == [False, True]
    assert diagnostics['lora_counts'].tolist() == [2.0, 2.0]
    assert diagnostics['forward_calls'].tolist() == [2.0, 2.0]
    expected = torch.tensor([
        [3.0, 1.0, 0.0],
        [3.0, 1.0, 0.0],
    ])
    assert torch.allclose(logits, expected)
