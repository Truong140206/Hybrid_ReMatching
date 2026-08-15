#!/usr/bin/env python3
"""Compare stage-by-task accuracy matrices from ImageNet-R evaluation logs."""

import argparse
import re
from pathlib import Path


TASK_END = re.compile(r'^Test: \[Task (\d+)\] Total time:')
TASK_PROGRESS = re.compile(r'^Test: \[Task (\d+)\]\s+\[')
STAGE_END = re.compile(r'\[Average accuracy till task(\d+)\]')
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def _summary_metrics(line):
    if not line.startswith('- Acc@task'):
        return None
    patterns = {
        'Acc@task': rf'Acc@task\s+({NUMBER})',
        'Acc@1': rf'Acc@1\s+({NUMBER})',
        'Acc@5': rf'Acc@5\s+({NUMBER})',
        'Loss': rf'[Ll]oss\s+({NUMBER})',
    }
    matches = {name: re.search(pattern, line)
               for name, pattern in patterns.items()}
    if not all(matches.values()):
        return None
    return {name: float(match.group(1))
            for name, match in matches.items()}


def _progress_metrics(line):
    patterns = {
        'Acc@task': rf'Acc@task:\s*{NUMBER}\s*\(({NUMBER})\)',
        'Acc@1': rf'Acc@1:\s*{NUMBER}\s*\(({NUMBER})\)',
        'Acc@5': rf'Acc@5:\s*{NUMBER}\s*\(({NUMBER})\)',
        'Loss': rf'Loss:\s*{NUMBER}\s*\(({NUMBER})\)',
    }
    matches = {name: re.search(pattern, line)
               for name, pattern in patterns.items()}
    if not all(matches.values()):
        return None
    return {name: float(match.group(1))
            for name, match in matches.items()}


def _normalize_stage_rows(stage_rows, stage):
    one_based = set(range(1, stage + 1))
    if one_based.issubset(stage_rows):
        return {task: dict(stage_rows[task]) for task in sorted(one_based)}

    zero_based = set(range(stage))
    if zero_based.issubset(stage_rows):
        return {task + 1: dict(stage_rows[task])
                for task in sorted(zero_based)}
    return None


def parse_accuracy_matrix(path):
    matrix = {}
    stage_rows = {}
    pending_task = None
    for raw_line in Path(path).read_text(
            encoding='utf-8', errors='replace').splitlines():
        line = raw_line.strip()

        progress_match = TASK_PROGRESS.match(line)
        if progress_match:
            progress_metrics = _progress_metrics(line)
            if progress_metrics is not None:
                stage_rows[int(progress_match.group(1))] = progress_metrics
            continue

        task_match = TASK_END.match(line)
        if task_match:
            pending_task = int(task_match.group(1))
            continue

        summary_metrics = _summary_metrics(line)
        if summary_metrics is not None and pending_task is not None:
            stage_rows[pending_task] = summary_metrics
            pending_task = None
            continue

        stage_match = STAGE_END.search(line)
        if stage_match:
            stage = int(stage_match.group(1))
            normalized = _normalize_stage_rows(stage_rows, stage)
            if normalized is not None:
                matrix[stage] = normalized
            stage_rows = {}
            pending_task = None

    if 10 not in matrix or set(matrix[10]) != set(range(1, 11)):
        available = {stage: sorted(rows) for stage, rows in matrix.items()}
        raise ValueError(
            f'Incomplete final 10-task matrix: {path}; parsed={available}')
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


def stage_average(matrix, stage, metric):
    rows = matrix[stage]
    return sum(row[metric] for row in rows.values()) / len(rows)


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


    common_stages = sorted(set(baseline_matrix) & set(proposal_matrix))
    lines.extend([
        '',
        '## Stagewise causal diagnosis',
        '',
        '| Stage | Base Acc@1 | Prop Acc@1 | ΔAcc@1 | Base Acc@5 | Prop Acc@5 | ΔAcc@5 | Base Loss | Prop Loss | ΔLoss |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for stage in common_stages:
        base_acc1 = stage_average(baseline_matrix, stage, 'Acc@1')
        prop_acc1 = stage_average(proposal_matrix, stage, 'Acc@1')
        base_acc5 = stage_average(baseline_matrix, stage, 'Acc@5')
        prop_acc5 = stage_average(proposal_matrix, stage, 'Acc@5')
        base_loss = stage_average(baseline_matrix, stage, 'Loss')
        prop_loss = stage_average(proposal_matrix, stage, 'Loss')
        lines.append(
            f'| {stage} | {base_acc1:.4f} | {prop_acc1:.4f} | '
            f'{prop_acc1 - base_acc1:+.4f} | '
            f'{base_acc5:.4f} | {prop_acc5:.4f} | '
            f'{prop_acc5 - base_acc5:+.4f} | '
            f'{base_loss:.4f} | {prop_loss:.4f} | '
            f'{prop_loss - base_loss:+.4f} |')

    lines.extend([
        '',
        '## Largest stage-task regressions',
        '',
    ])
    for metric, reverse in (('Acc@1', False), ('Acc@5', False),
                            ('Loss', True)):
        cells = []
        for stage in common_stages:
            common_tasks = (
                set(baseline_matrix[stage]) & set(proposal_matrix[stage]))
            for task in common_tasks:
                base_value = baseline_matrix[stage][task][metric]
                prop_value = proposal_matrix[stage][task][metric]
                cells.append((
                    prop_value - base_value, stage, task,
                    base_value, prop_value))
        cells.sort(reverse=reverse)
        lines.extend([f'### {metric}', ''])
        for delta, stage, task, base_value, prop_value in cells[:5]:
            lines.append(
                f'- Stage {stage}, task {task}: base {base_value:.4f}, '
                f'proposal {prop_value:.4f}, delta {delta:+.4f}.')
        lines.append('')

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
