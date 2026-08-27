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
    # Second copy of the statistics over an 80% subset of the current task, so
    # lambda can be chosen against the held-out 20% instead of being a constant
    # somebody tuned once. See _select_lambda.
    'gram_fit': None,
    'prototypes_fit': None,
    'count': 0,
    'count_fit': 0,
    'split_index': 0,
    'dim': None,
    'seen_classes': set(),
}


def reset_rp_head():
    _state['projection'] = None
    _state['gram'] = None
    _state['prototypes'] = None
    _state['gram_fit'] = None
    _state['prototypes_fit'] = None
    _state['count'] = 0
    _state['count_fit'] = 0
    _state['split_index'] = 0
    _state['dim'] = None
    _state['seen_classes'] = set()
    _state['temperature'] = None
    _state['weights'] = None
    _state['lambda_used'] = None
    # solve_rp_head always rewrites this, so a stale mask has never actually
    # leaked. Clearing it anyway: the invariant that keeps it safe lives in
    # another function, and a reset should leave nothing behind.
    _state['class_mask_bias'] = None


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
    features = _normalize(features.float(), args)
    # rp_dim <= 0 skips the projection entirely: plain ridge on the raw
    # features, which is the paper's "no RP" reference point.
    if int(getattr(args, 'rp_dim', 5000)) <= 0:
        return features
    projection = _ensure_projection(features.shape[1], device, args)
    kind = str(getattr(args, 'rp_activation', 'relu'))
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

    # Mirror the accumulation over a fixed 80% subset. The split is a running
    # index rather than a random draw so a run is reproducible without carrying
    # another generator, and because the caller walks the data class by class it
    # comes out stratified for free: every class contributes its own fifth.
    batch = projected.shape[0]
    index = torch.arange(_state['split_index'], _state['split_index'] + batch,
                         device=device)
    _state['split_index'] += batch
    _state['count'] += batch
    if not bool(getattr(args, 'rp_lambda_search', False)):
        return
    if _state['gram_fit'] is None:
        _state['gram_fit'] = torch.zeros(
            (dim, dim), dtype=torch.float64, device=device)
        _state['prototypes_fit'] = torch.zeros(
            (dim, nb_classes), dtype=torch.float64, device=device)
    keep = (index % 5) != 4
    if not bool(keep.any()):
        return
    fit_projected = projected[keep]
    _state['gram_fit'] += fit_projected.T @ fit_projected
    _state['prototypes_fit'] += fit_projected.T @ onehot[keep]
    _state['count_fit'] += int(keep.sum())


def _ridge(gram, lam):
    """G + lambda I, adding along the diagonal instead of building lam * I.

    At rp_dim=10000 a float64 identity is 800 MB and `lam * eye` materialises a
    second one, so the obvious form spends 2.4 GB of temporaries to express
    something that touches 10000 numbers. Arithmetically identical: every
    off-diagonal entry was gram + 0.0. This matters more now that the lambda
    search calls it seventeen times per task.
    """
    out = gram.clone()
    out.diagonal().add_(lam)
    return out


def _ridge_solve(gram, prototypes, lam):
    chol = torch.linalg.cholesky(_ridge(gram, lam))
    return torch.cholesky_solve(prototypes, chol)


@torch.no_grad()
def _select_lambda(args, device):
    """Choose lambda on held-out data from the current task, RanPAC's procedure.

    RanPAC does not ship a fixed lambda. At every task it splits that task's own
    data 80:20, sweeps lambda across 17 orders of magnitude, and keeps whichever
    value minimises squared error on the held-out fifth. Old tasks are never
    touched, so the continual-learning constraint holds. We had replaced that
    step with a constant tuned once on ImageNet-R seed 42 -- which is both a
    deviation from the method we build on and the single largest source of
    hyperparameter selection bias in the report.

    The validation error needs no stored features. With one-hot targets,

        ||Y - W^T H||_F^2 = n_val - 2 tr(W^T C_val) + tr(W^T G_val W)

    and the held-out statistics are just the difference between the full and the
    fitted accumulators. So the sweep is closed-form in quantities we already
    keep, and the exemplar-free protocol is preserved by construction rather
    than by argument.

    The sweep is the expensive part -- 17 Cholesky factorisations of a dim x dim
    matrix per task. RanPAC's own write-up names the same step as its bottleneck.
    """
    gram_fit = _state.get('gram_fit')
    if gram_fit is None:
        raise RuntimeError(
            'rp_lambda_search is on but no 80% statistics were accumulated; '
            'accumulate_rp_statistics must run with the same flag set')
    n_val = int(_state['count']) - int(_state['count_fit'])
    if n_val <= 0:
        raise RuntimeError('rp_lambda_search has an empty validation split')

    prototypes_fit = _state['prototypes_fit']
    gram_val = _state['gram'] - gram_fit
    prototypes_val = _state['prototypes'] - prototypes_fit

    best_lambda, best_error = None, None
    tried = []
    for exponent in range(-8, 9):
        lam = float(10.0 ** exponent)
        try:
            weights = _ridge_solve(gram_fit, prototypes_fit, lam)
        except torch.linalg.LinAlgError:
            # Tiny lambda on a rank-deficient Gram. Not a failure, just a value
            # the data cannot support; skip it rather than abort the sweep.
            continue
        error = (float(n_val)
                 - 2.0 * float((weights * prototypes_val).sum())
                 + float((weights * (gram_val @ weights)).sum()))
        tried.append((lam, error))
        if best_error is None or error < best_error:
            best_lambda, best_error = lam, error
        del weights
    del gram_val, prototypes_val

    if best_lambda is None:
        raise RuntimeError('rp_lambda_search: every lambda failed to solve')
    print('RP head lambda search: chose %.3g over %d values '
          '(val MSE %.6g, %d fit / %d val samples)'
          % (best_lambda, len(tried), best_error,
             _state['count_fit'], n_val))
    return best_lambda


@torch.no_grad()
def solve_rp_head(args, device, seen_class_ids=None):
    """Ridge solve  Wout = (G + lambda I)^-1 C  for the current statistics."""
    gram = _state['gram']
    prototypes = _state['prototypes']
    if gram is None:
        raise RuntimeError('RP head has no accumulated statistics')
    if bool(getattr(args, 'rp_lambda_search', False)):
        # Selected on 80%, but the final weights are refitted on everything.
        # Holding out a fifth is a device for picking lambda, not a reason to
        # throw that fifth away once lambda is fixed.
        lam = _select_lambda(args, device)
    else:
        lam = float(getattr(args, 'rp_lambda', 1e4))
    _state['lambda_used'] = lam
    ridge = _ridge(gram, lam)
    try:
        chol = torch.linalg.cholesky(ridge)
        weights = torch.cholesky_solve(prototypes, chol)
    except torch.linalg.LinAlgError as err:
        # Only the not-positive-definite case falls back. A bare `except
        # Exception` here also swallowed OOM, and swallowed it into lstsq,
        # which needs more memory than the Cholesky it was replacing. Say so
        # rather than degrading in silence.
        print('RP head: Cholesky failed (', err, ') - falling back to lstsq. '
              'A singular Gram at lambda =', lam, 'means the statistics are '
              'degenerate; check that accumulation actually ran.')
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
