#!/usr/bin/env python3
"""Compare stage-by-task accuracy matrices from ImageNet-R evaluation logs."""

import argparse
import re
from pathlib import Path


TASK_END = re.compile(r'^Test: \[Task (\d+)\] Total time:')
TASK_METRICS = re.compile(
    r'^- Acc@task\s+([-+0-9.]+)\s+Acc@1\s+([-+0-9.]+)\s+'
    r'Acc@5\s+([-+0-9.]+)\s+loss\s+([-+0-9.]+)')
STAGE_END = re.compile(r'\[Average accuracy till task(\d+)\]')


def parse_accuracy_matrix(path):
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
        metric_match = TASK_METRICS.match(line)
        if metric_match and pending_task is not None:
            stage_rows[pending_task] = {
                'Acc@task': float(metric_match.group(1)),
                'Acc@1': float(metric_match.group(2)),
                'Acc@5': float(metric_match.group(3)),
                'Loss': float(metric_match.group(4)),
            }
            pending_task = None
            continue
        stage_match = STAGE_END.search(line)
        if stage_match:
            stage = int(stage_match.group(1))
            expected = set(range(1, stage + 1))
            if expected.issubset(stage_rows):
                matrix[stage] = {
                    task: dict(stage_rows[task]) for task in sorted(expected)}
            stage_rows = {}
            pending_task = None

    if 10 not in matrix or set(matrix[10]) != set(range(1, 11)):
        raise ValueError(f'Incomplete final 10-task matrix: {path}')
    return matrix


def task_retention(matrix, metric='Acc@1'):
    final_stage = max(matrix)
    rows = {}
    for task in range(1, final_stage + 1):
        history = [matrix[stage][task][metric]
                   for stage in sorted(matrix)
                   if stage >= task and task in matrix[stage]]
        initial = matrix[task][task][metric]
        final = matrix[final_stage][task][metric]
        peak = max(history)
        rows[task] = {
            'initial': initial,
            'peak': peak,
            'final': final,
            'forgetting': peak - final,
            'backward': final - initial,
        }
    return rows


def average(rows, field, tasks):
    values = [rows[task][field] for task in tasks]
    return sum(values) / len(values)


def build_report(baseline_matrix, proposal_matrix, exhaustive_matrix=None):
    baseline = task_retention(baseline_matrix)
    proposal = task_retention(proposal_matrix)
    exhaustive = (task_retention(exhaustive_matrix)
                  if exhaustive_matrix is not None else None)

    lines = [
        '# ImageNet-R taskwise retention diagnostic',
        '',
        '| Task | Base initial | Prop initial | Base peak | Prop peak | Base final | Prop final | Δfinal | Base forget | Prop forget | Δforget |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for task in range(1, 11):
        base = baseline[task]
        prop = proposal[task]
        lines.append(
            f'| {task} | {base["initial"]:.4f} | {prop["initial"]:.4f} | '
            f'{base["peak"]:.4f} | {prop["peak"]:.4f} | '
            f'{base["final"]:.4f} | {prop["final"]:.4f} | '
            f'{prop["final"] - base["final"]:+.4f} | '
            f'{base["forgetting"]:.4f} | {prop["forgetting"]:.4f} | '
            f'{prop["forgetting"] - base["forgetting"]:+.4f} |')

    old_tasks = range(1, 10)
    lines.extend([
        '',
        '## Aggregate old-task diagnosis (tasks 1-9)',
        '',
        '| Quantity | Baseline | Proposal | Delta |',
        '|---|---:|---:|---:|',
    ])
    for field in ('initial', 'peak', 'final', 'forgetting', 'backward'):
        base_value = average(baseline, field, old_tasks)
        prop_value = average(proposal, field, old_tasks)
        lines.append(
            f'| {field} | {base_value:.4f} | {prop_value:.4f} | '
            f'{prop_value - base_value:+.4f} |')

    regressions = sorted(
        range(1, 10),
        key=lambda task: proposal[task]['final'] - baseline[task]['final'])
    lines.extend([
        '',
        '## Largest final Acc@1 regressions on old tasks',
        '',
    ])
    for task in regressions[:5]:
        delta = proposal[task]['final'] - baseline[task]['final']
        peak_delta = proposal[task]['peak'] - baseline[task]['peak']
        lines.append(
            f'- Task {task}: final delta {delta:+.4f}, '
            f'peak delta {peak_delta:+.4f}, '
            f'forgetting delta '
            f'{proposal[task]["forgetting"] - baseline[task]["forgetting"]:+.4f}.')

    if exhaustive is not None:
        lines.extend([
            '',
            '## Proposal headroom to exhaustive on final old-task Acc@1',
            '',
        ])
        gaps = sorted(
            range(1, 10),
            key=lambda task: proposal[task]['final'] - exhaustive[task]['final'])
        for task in gaps[:5]:
            lines.append(
                f'- Task {task}: proposal {proposal[task]["final"]:.4f}, '
                f'exhaustive {exhaustive[task]["final"]:.4f}, '
                f'gap {proposal[task]["final"] - exhaustive[task]["final"]:+.4f}.')

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--proposal', required=True, type=Path)
    parser.add_argument('--exhaustive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    report = build_report(
        parse_accuracy_matrix(args.baseline),
        parse_accuracy_matrix(args.proposal),
        parse_accuracy_matrix(args.exhaustive) if args.exhaustive else None,
    )
    if args.output:
        args.output.write_text(report, encoding='utf-8')
        print(f'Taskwise diagnostic: {args.output}')
    print(report, end='')


if __name__ == '__main__':
    main()
