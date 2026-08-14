import torch
from torch import nn

from engines.arrow_lora_audit import (
    _leading_input_direction,
    arrow_candidate_diagnostics,
    arrow_task_scores,
)
from peft.lora.hide_lora import HideLoraPool


class IdentityBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()

    def forward(self, x):
        return x


class TinyArrowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Identity()
        self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, 4))
        self.pos_drop = nn.Identity()
        self.blocks = nn.Sequential(IdentityBlock(), IdentityBlock())
        self.lora_layer = HideLoraPool(
            pool_size=2, depth=2, dim=4, rank=1)
        with torch.no_grad():
            for parameter in self.lora_layer.parameters():
                parameter.zero_()
            for depth in range(2):
                self.lora_layer.k_lora_A[0, depth, 0, 0] = 1.0
                self.lora_layer.k_lora_B[0, depth, 0, 0] = 1.0
                self.lora_layer.k_lora_A[1, depth, 1, 0] = 1.0
                self.lora_layer.k_lora_B[1, depth, 0, 1] = 1.0


def test_leading_direction_matches_lora_input_axis():
    k_a = torch.tensor([[0.0], [3.0], [0.0], [0.0]])
    k_b = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    zero_a = torch.zeros_like(k_a)
    zero_b = torch.zeros_like(k_b)

    direction = _leading_input_direction(k_a, k_b, zero_a, zero_b)

    assert torch.allclose(
        direction.abs(), torch.tensor([0., 1., 0., 0.]))


def test_layerwise_scores_select_matching_adapter_without_parameter_changes():
    model = TinyArrowModel().eval()
    before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }
    inputs = torch.tensor([
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    ])

    scores = arrow_task_scores(model, inputs, seen_task_count=2)

    assert scores.argmax(dim=1).tolist() == [1, 0]
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])


def test_tii_arrow_union_recovers_complementary_winners():
    tii_ranking = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    arrow_ranking = torch.tensor([[3, 2, 1, 0], [2, 3, 1, 0]])
    winners = torch.tensor([3, 1])

    diagnostics = arrow_candidate_diagnostics(
        tii_ranking, arrow_ranking, winners)

    assert diagnostics['arrow_union_recall_2x2'].tolist() == [True, True]
    assert diagnostics['arrow_union_lora_counts'].tolist() == [4.0, 4.0]
    assert diagnostics['arrow_winner_recall_2'].tolist() == [True, False]
