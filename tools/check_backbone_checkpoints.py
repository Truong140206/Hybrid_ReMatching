#!/usr/bin/env python3
"""Verify -- and where needed convert -- the self-supervised backbone weights.

Why this exists. All three loaders in vits/hrm_lora_vision_transformer.py end
with the same three lines:

    not_in_k = [k for k in ckpt.keys() if k not in state_dict.keys()]
    for k in not_in_k: del ckpt[k]
    state_dict.update(ckpt); model.load_state_dict(state_dict)

If the checkpoint's parameter names do not match the model's, every key is
deleted, `update` runs on an empty dict, and the model keeps its RANDOM
initialisation. load_state_dict then succeeds, because the state dict it is
handed is the model's own. No exception, no warning -- just a randomly
initialised backbone that trains for hours and produces nonsense.

The MoCo v3 file published by Facebook is exactly that case: the loader reads
ckpt['model'] with timm-style names, while the download has ckpt['state_dict']
with keys prefixed `module.base_encoder.`. So it is converted here rather than
discovered later.

Usage:  python tools/check_backbone_checkpoints.py [--convert]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vits.hrm_lora_vision_transformer import _create_vision_transformer  # noqa: E402

CKPT_DIR = 'checkpoints'

# name -> (filename, top-level key the loader reads, prefix the loader strips)
TARGETS = [
    ('iBOT-1K',   'checkpoint_teacher.pth',        'state_dict', ''),
    ('iBOT-21K',  'checkpoint.pth',                'teacher',    'backbone.'),
    ('MoCo v3',   'mocov3-vit-base-300ep.pth',     'model',      ''),
    ('DINO-1K',   'dino_vitbase16_pretrain.pth',  None,       ''),
]


def reference_state_dict():
    """Same construction the loaders use, so the comparison is the real one."""
    model = _create_vision_transformer(
        'vit_base_patch16_224_in21k', pretrained=False,
        patch_size=16, embed_dim=768, depth=12, num_heads=12)
    return model.state_dict()


def convert_mocov3(path, reference):
    """Facebook's file -> what the loader expects, or explain why it cannot."""
    raw = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(raw, dict) and 'model' in raw:
        print('    already has a "model" key; leaving it alone')
        return False
    if not (isinstance(raw, dict) and 'state_dict' in raw):
        print('    cannot convert: no "state_dict" key, found %s'
              % list(raw)[:8])
        return False
    out = {}
    for key, value in raw['state_dict'].items():
        name = key
        for prefix in ('module.base_encoder.', 'module.', 'base_encoder.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        out[name] = value
    hit = sum(1 for k in out if k in reference)
    print('    converted: %d tensors, %d match the model' % (len(out), hit))
    if hit < 100:
        print('    REFUSING to save: too few matches, the naming is not what '
              'this loader expects')
        return False
    torch.save({'model': out}, path)
    print('    saved back to %s under a "model" key' % path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert', action='store_true',
                        help='rewrite the MoCo v3 file into the expected form')
    args = parser.parse_args()

    reference = reference_state_dict()
    print('model has %d parameter tensors\n' % len(reference))

    ok = True
    for label, filename, top_key, strip in TARGETS:
        path = os.path.join(CKPT_DIR, filename)
        print('%s  (%s)' % (label, path))
        if not os.path.exists(path):
            print('    MISSING\n')
            ok = False
            continue

        if label.startswith('MoCo') and args.convert:
            convert_mocov3(path, reference)

        raw = torch.load(path, map_location='cpu', weights_only=False)
        if top_key is not None and (
                not isinstance(raw, dict) or top_key not in raw):
            print('    FAIL: loader reads ["%s"], file has %s\n'
                  % (top_key, list(raw)[:8] if isinstance(raw, dict) else type(raw)))
            ok = False
            continue

        ckpt = raw if top_key is None else raw[top_key]
        renamed = ({k.replace(strip, ''): v for k, v in ckpt.items()}
                   if strip else dict(ckpt))
        hit = sum(1 for k in renamed if k in reference)
        missing = sum(1 for k in reference if k not in renamed)
        verdict = 'OK' if hit >= 100 else 'FAIL -- would load RANDOM weights'
        if hit < 100:
            ok = False
        print('    %d tensors, %d match, %d model tensors uncovered  -> %s\n'
              % (len(renamed), hit, missing, verdict))

    print('ALL GOOD' if ok else 'SOMETHING IS WRONG -- do not start training')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
