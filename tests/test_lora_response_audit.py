import torch
from torch import nn

from engines.lora_response_audit import (
    lora_response_candidate_diagnostics,
    lora_response_task_scores,
)
from peft.lora.hide_lora import HideLoraPool


class IdentityBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.Identity()

    def forward(self, x):
        return x


class TinyResponseModel(nn.Module):
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
                self.lora_layer.k_lora_B[0, depth, 0, 0] = 10.0
                self.lora_layer.k_lora_A[1, depth, 1, 0] = 1.0
                self.lora_layer.k_lora_B[1, depth, 0, 1] = 1.0


def test_full_rank_response_selects_matching_subspace_not_largest_adapter():
    model = TinyResponseModel().eval()
    inputs = torch.tensor([
        [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    ])

    scores = lora_response_task_scores(model, inputs, seen_task_count=2)

    assert scores.argmax(dim=1).tolist() == [1, 0]
    assert torch.allclose(scores[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(scores[1], torch.tensor([1.0, 0.0]))


def test_response_union_recovers_complementary_winners():
    tii_ranking = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    response_ranking = torch.tensor([[3, 2, 1, 0], [2, 3, 1, 0]])
    winners = torch.tensor([3, 1])

    diagnostics = lora_response_candidate_diagnostics(
        tii_ranking, response_ranking, winners)

    assert diagnostics['response_union_recall_2x2'].tolist() == [True, True]
    assert diagnostics['response_union_lora_counts'].tolist() == [4.0, 4.0]
    assert diagnostics['response_winner_recall_2'].tolist() == [True, False]
