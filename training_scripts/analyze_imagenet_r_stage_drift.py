#!/usr/bin/env python3
"""Decompose ImageNet-R retention decay into local, competition, and routing terms."""

import argparse
import re
from pathlib import Path

try:
    from training_scripts.compare_imagenet_r_taskwise import parse_accuracy_matrix
except ModuleNotFoundError:
    from compare_imagenet_r_taskwise import parse_accuracy_matrix


TASK_END = re.compile(r'^Test: \[Task (\d+)\] Total time:')
STAGE_END = re.compile(r'\[Average accuracy till task(\d+)\]')
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
METRICS = (
    'OwnLocalAcc@1',
    'OwnSeenAcc@1',
    'OwnLocalLoss',
    'OwnSeenLoss',
    'OwnSeenTaskAcc',
    'LocalToSeenFailure',
)


def _stage_drift_metrics(line):
    if not line.startswith('* StageDriftAudit '):
        return None
    values = {}
    for name in METRICS:
        match = re.search(rf'{re.escape(name)}\s+({NUMBER})', line)
        if not match:
            raise ValueError(f'Missing {name} in stage-drift row: {line}')
        values[name] = float(match.group(1))
    return values


def parse_stage_drift_matrix(path):
    matrix = {}
    stage_rows = {}
    pending_task = None
    for raw_line in Path(path).read_text(
            encoding='utf-8', errors='replace').splitlines():
        line = raw_line.strip()
        task_match = TASK_END.match(line)
        if task_match:
            pending_task = int(task_match.group(1))
            continue
        metrics = _stage_drift_metrics(line)
        if metrics is not None:
            if pending_task is None:
                raise ValueError(
                    f'Stage-drift row has no preceding task row in {path}')
            stage_rows[pending_task] = metrics
            continue
        stage_match = STAGE_END.search(line)
        if stage_match:
            stage = int(stage_match.group(1))
            expected = set(range(1, stage + 1))
            if expected.issubset(stage_rows):
                matrix[stage] = {
                    task: dict(stage_rows[task]) for task in sorted(expected)
                }
            stage_rows = {}
            pending_task = None
    if 10 not in matrix or set(matrix[10]) != set(range(1, 11)):
        available = {stage: sorted(rows) for stage, rows in matrix.items()}
        raise ValueError(
            f'Incomplete final stage-drift matrix: {path}; parsed={available}')
    return matrix


def _average(values):
    values = list(values)
    return sum(values) / len(values)


def build_stage_drift_report(drift_matrix, proposal_matrix):
    final_stage = max(drift_matrix)
    old_tasks = range(1, final_stage)
    lines = [
        '# ImageNet-R stage-drift decomposition',
        '',
        '| Task | Local initial | Local final | Δlocal | Seen initial | '
        'Seen final | Δseen | Cross-gap growth | Final proposal | '
        'Proposal-vs-seen |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    rows = {}
    for task in range(1, final_stage + 1):
        initial = drift_matrix[task][task]
        final = drift_matrix[final_stage][task]
        proposal_final = proposal_matrix[final_stage][task]['Acc@1']
        initial_gap = initial['OwnLocalAcc@1'] - initial['OwnSeenAcc@1']
        final_gap = final['OwnLocalAcc@1'] - final['OwnSeenAcc@1']
        row = {
            'local_delta': (
                final['OwnLocalAcc@1'] - initial['OwnLocalAcc@1']),
            'seen_delta': (
                final['OwnSeenAcc@1'] - initial['OwnSeenAcc@1']),
            'cross_gap_growth': final_gap - initial_gap,
            'proposal_vs_seen': proposal_final - final['OwnSeenAcc@1'],
            'local_loss_delta': (
                final['OwnLocalLoss'] - initial['OwnLocalLoss']),
            'seen_loss_delta': (
                final['OwnSeenLoss'] - initial['OwnSeenLoss']),
        }
        rows[task] = row
        lines.append(
            f'| {task} | {initial["OwnLocalAcc@1"]:.4f} | '
            f'{final["OwnLocalAcc@1"]:.4f} | {row["local_delta"]:+.4f} | '
            f'{initial["OwnSeenAcc@1"]:.4f} | '
            f'{final["OwnSeenAcc@1"]:.4f} | {row["seen_delta"]:+.4f} | '
            f'{row["cross_gap_growth"]:+.4f} | {proposal_final:.4f} | '
            f'{row["proposal_vs_seen"]:+.4f} |')

    local_decay = _average(-rows[t]['local_delta'] for t in old_tasks)
    competition_growth = _average(
        rows[t]['cross_gap_growth'] for t in old_tasks)
    routing_penalty = _average(
        -rows[t]['proposal_vs_seen'] for t in old_tasks)
    local_loss_growth = _average(
        rows[t]['local_loss_delta'] for t in old_tasks)
    seen_loss_growth = _average(
        rows[t]['seen_loss_delta'] for t in old_tasks)

    signals = {
        'TRUE_ADAPTER_LOCAL_DRIFT': local_decay,
        'CROSS_TASK_SCORE_COMPETITION': competition_growth,
        'ROUTING_SELECTION_PENALTY': routing_penalty,
    }
    dominant = max(signals, key=signals.get)
    ranked = sorted(signals.items(), key=lambda item: item[1], reverse=True)

    lines.extend([
        '',
        '## Aggregate old-task decomposition (tasks 1-9)',
        '',
        f'- True-adapter local accuracy decay: {local_decay:+.4f}.',
        f'- Cross-task competition gap growth: {competition_growth:+.4f}.',
        f'- Routing penalty versus true adapter over seen classes: '
        f'{routing_penalty:+.4f}.',
        f'- True-adapter local loss growth: {local_loss_growth:+.4f}.',
        f'- True-adapter seen-class loss growth: {seen_loss_growth:+.4f}.',
        f'- Dominant measured signal: {dominant}.',
        '',
        'Signal ranking:',
        '',
    ])
    for name, value in ranked:
        lines.append(f'- {name}: {value:+.4f}')

    focus = (6, 7, 8)
    lines.extend([
        '',
        '## Focus tasks 6-8',
        '',
        f'- Mean local delta: '
        f'{_average(rows[t]["local_delta"] for t in focus):+.4f}.',
        f'- Mean seen delta: '
        f'{_average(rows[t]["seen_delta"] for t in focus):+.4f}.',
        f'- Mean cross-gap growth: '
        f'{_average(rows[t]["cross_gap_growth"] for t in focus):+.4f}.',
        f'- Mean proposal-vs-seen: '
        f'{_average(rows[t]["proposal_vs_seen"] for t in focus):+.4f}.',
        '',
        '## Locked next-action map',
        '',
        '- TRUE_ADAPTER_LOCAL_DRIFT: change training/CRCT stability; do not '
        'tune routing.',
        '- CROSS_TASK_SCORE_COMPETITION: use stage-consistent cross-task '
        'scoring constrained to preserve within-task ranking.',
        '- ROUTING_SELECTION_PENALTY: improve candidate selection while '
        'keeping the current 3.0529-LoRA consensus budget.',
        '',
        'These are diagnostic magnitudes, not a quality gate and not additive '
        'causal effects.',
    ])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = build_stage_drift_report(
        parse_stage_drift_matrix(args.log),
        parse_accuracy_matrix(args.log),
    )
    if args.output:
        args.output.write_text(report, encoding='utf-8')
        print(f'Stage-drift diagnostic: {args.output}')
    print(report, end='')


if __name__ == '__main__':
    main()