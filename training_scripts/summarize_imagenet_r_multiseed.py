#!/usr/bin/env python3
"""Summarize locked ImageNet-R baseline/exhaustive/proposal runs across seeds."""

import argparse
import re
import statistics
from pathlib import Path


METRICS = ('Acc@task', 'Acc@1', 'Acc@5', 'Loss', 'Forgetting', 'Backward')
METHODS = ('Baseline', 'Exhaustive', 'Proposal')
WALL_TIME_PATTERNS = {
    'Baseline': r'Conventional evaluation wall time seconds:\s*(\d+)',
    'Exhaustive': r'Vectorized exhaustive wall time seconds:\s*(\d+)',
    'Proposal': r'Prediction-proposal evaluation wall time seconds:\s*(\d+)',
}


def parse_final_metrics(path):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    rows = [line.strip() for line in text.splitlines()
            if 'Average accuracy till task10' in line]
    if not rows:
        raise ValueError(f'Missing task-10 metrics: {path}')
    row = rows[-1]
    values = {}
    for name in METRICS:
        match = re.search(
            rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
        if match is None:
            raise ValueError(f'Missing {name} in {path}')
        values[name] = float(match.group(1))
    for name in ('LoRA/sample', 'ForwardCalls/sample'):
        match = re.search(
            rf'{re.escape(name)}:\s*(-?[0-9]+(?:\.[0-9]+)?)', row)
        values[name] = float(match.group(1)) if match else None
    return values, text


def parse_wall_time(text, method):
    pattern = WALL_TIME_PATTERNS.get(method)
    if pattern is None:
        return None
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


def run_paths(output_root, run_template, seed):
    run_name = run_template.format(seed=seed)
    base = Path(output_root)
    return {
        'Baseline': base / f'{run_name}_eval_conventional.log',
        'Exhaustive': base / (
            f'{run_name}_eval_vectorized_exhaustive_c4_p0p3_t1p0.log'),
        'Proposal': base / (
            f'{run_name}_eval_prediction_proposal_i2_p3_c5_tiicomplete_strict.log'),
    }


def collect_runs(output_root, seeds, run_template):
    records = []
    missing = []
    for seed in seeds:
        for method, path in run_paths(output_root, run_template, seed).items():
            if not path.is_file():
                missing.append(str(path))
                continue
            metrics, text = parse_final_metrics(path)
            records.append({
                'seed': int(seed),
                'method': method,
                'path': path,
                'wall_time': parse_wall_time(text, method),
                **metrics,
            })
    if missing:
        raise FileNotFoundError(
            'Missing required logs:\n  ' + '\n  '.join(missing))
    return records


def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def fmt(value, digits=4):
    return '-' if value is None else f'{value:.{digits}f}'


def build_markdown(records, seeds):
    lines = [
        '# ImageNet-R multi-seed summary',
        '',
        'Cấu hình được khóa trước khi chạy nhiều seed; std là sample standard deviation.',
        '',
        '| Seed | Method | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | LoRA/mẫu | Calls/mẫu | Time (s) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for record in records:
        lines.append(
            f'| {record["seed"]} | {record["method"]} | '
            + ' | '.join(fmt(record[name]) for name in METRICS)
            + f' | {fmt(record["LoRA/sample"])}'
            + f' | {fmt(record["ForwardCalls/sample"])}'
            + f' | {record["wall_time"] if record["wall_time"] is not None else "-"} |')

    lines.extend([
        '',
        '## Mean ± std',
        '',
        '| Method | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | Time (s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for method in METHODS:
        subset = [record for record in records if record['method'] == method]
        cells = []
        for name in METRICS:
            mean, std = mean_std([record[name] for record in subset])
            cells.append(f'{mean:.4f} ± {std:.4f}')
        times = [record['wall_time'] for record in subset
                 if record['wall_time'] is not None]
        if len(times) == len(seeds):
            mean, std = mean_std(times)
            time_cell = f'{mean:.1f} ± {std:.1f}'
        else:
            time_cell = '-'
        lines.append(f'| {method} | ' + ' | '.join(cells) + f' | {time_cell} |')

    lines.extend([
        '',
        '## Proposal so với baseline theo từng seed',
        '',
        '| Seed | ΔAcc@task | ΔAcc@1 | ΔAcc@5 | ΔLoss | ΔForgetting | ΔBackward | All-quality-pass |',
        '|---:|---:|---:|---:|---:|---:|---:|:---:|',
    ])
    for seed in seeds:
        by_method = {
            record['method']: record for record in records
            if record['seed'] == int(seed)
        }
        baseline = by_method['Baseline']
        proposal = by_method['Proposal']
        deltas = {name: proposal[name] - baseline[name] for name in METRICS}
        passed = (
            all(deltas[name] >= 0.0 for name in ('Acc@task', 'Acc@1', 'Acc@5', 'Backward'))
            and all(deltas[name] <= 0.0 for name in ('Loss', 'Forgetting'))
        )
        lines.append(
            f'| {seed} | ' + ' | '.join(
                f'{deltas[name]:+.4f}' for name in METRICS
            ) + f' | {"PASS" if passed else "FAIL"} |')

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', required=True, type=int)
    parser.add_argument(
        '--run-template',
        default='imr_lora_rank8_baseline_10tasks_seed{seed}')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    records = collect_runs(args.output_root, args.seeds, args.run_template)
    report = build_markdown(records, args.seeds)
    if args.output:
        args.output.write_text(report, encoding='utf-8')
        print(f'Multi-seed summary: {args.output}')
    print(report, end='')


if __name__ == '__main__':
    main()
