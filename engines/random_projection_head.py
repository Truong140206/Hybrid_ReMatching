"""Routing-free second-order classifier head (RanPAC-style).

Motivation
----------
HRM-PET routes every query to one task LoRA before classifying, so its accuracy
is capped by task-inference quality. Measurements on Split-CUB200 showed that
ceiling is hard: exhaustive re-matching, Gaussian re-scoring and doubling the
LoRA rank all failed to move Acc@1 past ~86.5. Published exemplar-free results
(RanPAC, CUB B0I10) reach ~90 without any per-task routing, by classifying in a
high-dimensional random feature space with decorrelated second-order statistics.

This module implements that head:

    h = phi(f @ W)                       frozen random projection + nonlinearity
    G = sum_i h_i h_i^T                  Gram matrix  (M x M)
    C = sum_i h_i onehot(y_i)^T          class prototypes (M x nb_classes)
    s = h_test^T (G + lambda I)^-1 C     ridge / LDA-style scores

Only the aggregates G and C are retained -- no per-example features and no
images -- so the strict exemplar-free protocol is preserved (same category as
the class mean/covariance HRM-PET already stores). The projection is generated
from a fixed seed, so it never needs to be stored either.

The Gram inverse decorrelates the expanded features, which is what a per-class
diagonal/full covariance in the raw 768-d space failed to do.
"""

import math

import torch


_state = {
    'projection': None,
    'gram': None,
    'prototypes': None,
    'dim': None,
    'seen_classes': set(),
}


def reset_rp_head():
    _state['projection'] = None
    _state['gram'] = None
    _state['prototypes'] = None
    _state['dim'] = None
    _state['seen_classes'] = set()
    _state['temperature'] = None
    _state['weights'] = None


def rp_head_ready():
    return _state['gram'] is not None and bool(_state['seen_classes'])


def get_rp_state():
    return _state


def _activation(x, kind):
    if kind == 'relu':
        return torch.relu(x)
    if kind == 'square':
        return x * x
    if kind == 'gelu':
        return torch.nn.functional.gelu(x)
    if kind == 'none':
        return x
    raise ValueError('Unknown rp_activation: {}'.format(kind))


def _ensure_projection(feature_dim, device, args):
    """Frozen random projection, regenerated deterministically from a seed."""
    if _state['projection'] is not None:
        return _state['projection']
    dim = int(getattr(args, 'rp_dim', 5000))
    seed = int(getattr(args, 'rp_seed', 1993))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    projection = torch.randn(
        feature_dim, dim, generator=generator, dtype=torch.float32)
    projection = projection.to(device)
    _state['projection'] = projection
    _state['dim'] = dim
    return projection


def _normalize(features, args):
    """Feature scaling before projection.

    Raw ViT pre_logits have a large and uneven scale, which distorts a ReLU
    random projection and makes a single ridge lambda fit all classes badly.
    """
    kind = str(getattr(args, 'rp_normalize', 'none'))
    if kind == 'none':
        return features
    if kind == 'l2':
        return torch.nn.functional.normalize(features, dim=1)
    if kind == 'scale':
        return features / features.norm(dim=1, keepdim=True).mean().clamp_min(1e-6)
    raise ValueError('Unknown rp_normalize: {}'.format(kind))


def project_features(features, args, device):
    projection = _ensure_projection(features.shape[1], device, args)
    kind = str(getattr(args, 'rp_activation', 'relu'))
    features = _normalize(features.float(), args)
    return _activation(features @ projection, kind)


@torch.no_grad()
def accumulate_rp_statistics(features, targets, args, device, nb_classes):
    """Add one batch of training features to the Gram matrix and prototypes."""
    projected = project_features(features, args, device)
    dim = projected.shape[1]
    if _state['gram'] is None:
        _state['gram'] = torch.zeros(
            (dim, dim), dtype=torch.float64, device=device)
        _state['prototypes'] = torch.zeros(
            (dim, nb_classes), dtype=torch.float64, device=device)
    projected = projected.double()
    _state['gram'] += projected.T @ projected
    onehot = torch.zeros(
        (projected.shape[0], nb_classes), dtype=torch.float64, device=device)
    onehot[torch.arange(projected.shape[0], device=device),
           targets.to(device).long()] = 1.0
    _state['prototypes'] += projected.T @ onehot
    _state['seen_classes'].update(int(t) for t in targets.detach().cpu().tolist())


@torch.no_grad()
def solve_rp_head(args, device, seen_class_ids=None):
    """Ridge solve  Wout = (G + lambda I)^-1 C  for the current statistics."""
    gram = _state['gram']
    prototypes = _state['prototypes']
    if gram is None:
        raise RuntimeError('RP head has no accumulated statistics')
    lam = float(getattr(args, 'rp_lambda', 1e4))
    dim = gram.shape[0]
    ridge = gram + lam * torch.eye(dim, dtype=torch.float64, device=device)
    try:
        chol = torch.linalg.cholesky(ridge)
        weights = torch.cholesky_solve(prototypes, chol)
    except Exception:
        weights = torch.linalg.lstsq(ridge, prototypes).solution
    weights = weights.float()
    if seen_class_ids is not None:
        mask = torch.full((weights.shape[1],), float('-inf'), device=device)
        mask[torch.as_tensor(
            sorted(int(c) for c in seen_class_ids), device=device)] = 0.0
        _state['class_mask_bias'] = mask
    else:
        _state['class_mask_bias'] = None
    _state['weights'] = weights
    return weights


def fit_rp_temperature(scores, targets, args, device):
    """Fit one scalar temperature so the ridge scores read as calibrated logits.

    Ridge outputs are unnormalized regression values, so a cross-entropy loss
    computed on them is not comparable to a softmax classifier's loss. Only the
    scalar is kept; the scores used to fit it are transient.
    """
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=60)
    # Unseen classes carry a -inf mask bias; differentiating through it yields
    # NaN gradients, so fit on finite scores.
    scores = torch.nan_to_num(
        scores.detach().to(device).float(),
        nan=0.0, posinf=1e4, neginf=-1e4)
    targets = targets.detach().to(device)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            scores / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().item())
    if not math.isfinite(temperature) or temperature <= 0.0:
        print('RP head calibration produced', temperature,
              '- falling back to temperature 1.0')
        temperature = 1.0
    _state['temperature'] = temperature
    return temperature


@torch.no_grad()
def rp_head_predict(features, args, device):
    """Score every class for a batch of backbone features (no routing)."""
    weights = _state.get('weights')
    if weights is None:
        raise RuntimeError('RP head weights not solved yet')
    projected = project_features(features, args, device)
    scores = projected @ weights
    temperature = _state.get('temperature')
    if temperature and math.isfinite(temperature) and temperature > 0.0:
        scores = scores / temperature
    bias = _state.get('class_mask_bias')
    if bias is not None:
        scores = scores + bias
    return scores
