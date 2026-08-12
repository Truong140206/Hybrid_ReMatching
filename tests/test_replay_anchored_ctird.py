from types import SimpleNamespace

import torch
from torch import nn

from engines.replay_anchored_ctird import (
    ReplayAnchorMemory,
    invert_cfs_features,
    replay_anchor_relation_loss,
)


class TinyPatchEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)

    def forward(self, images):
        tokens = self.proj(images)
        return tokens.flatten(2).transpose(1, 2)


class TinyBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens, **unused):
        return tokens + 0.1 * torch.tanh(self.projection(tokens))


class TinyVisionModel(nn.Module):
    def __init__(self, image_size=8, embed_dim=6, classes=2):
        super().__init__()
        self.patch_embed = TinyPatchEmbed(embed_dim)
        patch_count = (image_size // 4) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList([TinyBlock(embed_dim), TinyBlock(embed_dim)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, classes)
        self.depth = len(self.blocks)
        self.lora_depth = 0
        self.lora_layer = None
        self.class_token = True
        self.global_pool = 'token'

    def forward(self, images, task_id=-1, train=False):
        tokens = self.patch_embed(images)
        tokens = torch.cat((self.cls_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        features = self.norm(tokens)[:, 0]
        return {
            'pre_logits': features,
            'features': features,
            'logits': self.head(features),
        }


def _args(tmp_path):
    return SimpleNamespace(
        input_size=8,
        replay_inversion_split_block=1,
        replay_inversion_layer_steps=2,
        replay_inversion_full_steps=2,
        replay_inversion_layer_lr=0.05,
        replay_inversion_full_lr=0.01,
        replay_inversion_class_weight=0.1,
        replay_inversion_tv_weight=0.0005,
        replay_anchor_batch_size=4,
        replay_anchor_teacher_confidence=0.0,
        replay_anchor_temperature=1.0,
        output_dir=str(tmp_path),
        replay_anchor_cache_dir=str(tmp_path),
    )


def test_inversion_returns_cache_ready_uint8_images(tmp_path):
    torch.manual_seed(7)
    model = TinyVisionModel()
    target_images = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        targets = model(target_images)['pre_logits']

    images = invert_cfs_features(
        model=model,
        target_features=targets,
        class_id=0,
        task_id=0,
        args=_args(tmp_path),
        device=torch.device('cpu'),
    )

    assert images.shape == (2, 3, 8, 8)
    assert images.dtype == torch.uint8


def test_replay_relation_loss_uses_cached_teacher_images(tmp_path):
    torch.manual_seed(11)
    model = TinyVisionModel(classes=1)
    torch.save(
        {
            'version': 1,
            'images': torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8),
            'class_id': 0,
            'task_id': 0,
        },
        tmp_path / 'class_0000.pth',
    )
    memory = ReplayAnchorMemory(str(tmp_path), [0])

    loss, kept, confidence = replay_anchor_relation_loss(
        model=model,
        memory=memory,
        current_task_id=1,
        seen_classes=[0],
        args=_args(tmp_path),
        device=torch.device('cpu'),
    )

    assert kept == 4
    assert confidence == 1.0
    assert torch.isfinite(loss)
