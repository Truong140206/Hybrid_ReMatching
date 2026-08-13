"""Protocol validation for rehearsal-free continual-learning experiments."""


_FORBIDDEN_BOOLEAN_FLAGS = {
    'crct_real_feature_replay': 'stores per-example real features',
    'prototype_rematching': 'uses stored real-feature prototypes',
    'shared_prototype_router': 'reconstructs prototypes from historical train images',
    'replay_logit_calibration': 'fits calibration on retained real features',
    'replay_task_router': 'trains a router on retained real features',
    'calibrated_progressive_rematching': 'trains halting gates on historical train images',
    'distilled_router_rematching': 'trains a router on historical train images',
}


def exemplar_free_violations(args):
    violations = []
    for name, reason in _FORBIDDEN_BOOLEAN_FLAGS.items():
        if bool(getattr(args, name, False)):
            violations.append(f'--{name}: {reason}')
    if float(getattr(args, 'exhaustive_local_prototype_weight', 0.0)) > 0.0:
        violations.append(
            '--exhaustive_local_prototype_weight: uses stored real-feature prototypes')
    return violations


def validate_exemplar_free_protocol(args):
    if not bool(getattr(args, 'strict_exemplar_free', False)):
        return
    violations = exemplar_free_violations(args)
    if violations:
        details = '\n  - '.join(violations)
        raise ValueError(
            'Strict exemplar-free protocol rejected this configuration:\n  - '
            + details)
    print(
        'Strict exemplar-free protocol: PASS '
        '(no historical images or per-example real features).')
