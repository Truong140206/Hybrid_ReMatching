import torch
from torch import nn

from engines.cfs_pmi_diagnostic import (
    images_to_patch_tokens,
    partial_invert_feature_targets,
    patch_tokens_to_output,
)


class TinyPatchEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)

    def forward(self, images):
        return self.proj(images).flatten(2).transpose(1, 2)


class TinyMixingBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.local = nn.Linear(embed_dim, embed_dim)
        self.context = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens, **unused):
        context = self.context(tokens.mean(dim=1, keepdim=True))
        return tokens + 0.1 * torch.tanh(self.local(tokens) + context)


class TinyVisionModel(nn.Module):
    def __init__(self, image_size=8, embed_dim=8, classes=2):
        super().__init__()
        self.patch_embed = TinyPatchEmbed(embed_dim)
        patch_count = (image_size // 4) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList([
            TinyMixingBlock(embed_dim),
            TinyMixingBlock(embed_dim),
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Identity()
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(embed_dim, classes)
        self.depth = len(self.blocks)
        self.lora_depth = 0
        self.lora_layer = None
        self.class_token = True
        self.global_pool = 'token'

    def forward(self, images, task_id=-1, train=False):
        tokens = self.patch_embed(images)
        tokens = torch.cat(
            (self.cls_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        for block in self.blocks:
            tokens = block(tokens)
        pre_logits = self.norm(tokens)[:, 0]
        classifier_features = self.fc_norm(self.mlp(pre_logits))
        return {
            'pre_logits': pre_logits,
            'logits': self.head(classifier_features),
        }


def test_patch_token_forward_matches_normal_forward():
    torch.manual_seed(3)
    model = TinyVisionModel().eval()
    images = torch.rand(3, 3, 8, 8)

    direct = model(images, task_id=0)
    tokens = images_to_patch_tokens(model, images)
    partial = patch_tokens_to_output(model, tokens, task_id=0)

    assert torch.allclose(direct['pre_logits'], partial['pre_logits'])
    assert torch.allclose(direct['logits'], partial['logits'])


def test_partial_inversion_improves_alignment_and_preserves_model():
    torch.manual_seed(5)
    model = TinyVisionModel().eval()
    reference_images = torch.rand(12, 3, 8, 8)
    target_images = torch.rand(2, 3, 8, 8)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        reference_tokens = images_to_patch_tokens(model, reference_images)
        targets = model(target_images, task_id=0)['pre_logits']
        initial_tokens = reference_tokens[:2].clone()
        initial_output = patch_tokens_to_output(model, initial_tokens, task_id=0)
        initial_cosine = torch.cosine_similarity(
            initial_output['pre_logits'], targets, dim=1).mean()

    _, output = partial_invert_feature_targets(
        model=model,
        target_features=targets,
        labels=torch.zeros(2, dtype=torch.long),
        task_id=0,
        seen_classes=[0, 1],
        token_mean=reference_tokens.mean(dim=0),
        token_var=reference_tokens.var(dim=0, unbiased=False),
        split_block=1,
        layer_steps=10,
        full_steps=20,
        class_weight=0.0,
        moment_weight=0.001,
    )
    final_cosine = torch.cosine_similarity(
        output['pre_logits'], targets, dim=1).mean()

    assert torch.isfinite(output['pre_logits']).all()
    assert final_cosine >= initial_cosine
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters_before[name])
        assert parameter.requires_grad
