#!/usr/bin/env python3
import re
import sys
from pathlib import Path


METRIC_PATTERN = re.compile(
    r'Acc@task:\s*([+-]?\d+(?:\.\d+)?)\s+'
    r'Acc@1:\s*([+-]?\d+(?:\.\d+)?)\s+'
    r'Acc@5:\s*([+-]?\d+(?:\.\d+)?)\s+'
    r'Loss:\s*([+-]?\d+(?:\.\d+)?)\s+'
    r'Forgetting:\s*([+-]?\d+(?:\.\d+)?)\s+'
    r'Backward:\s*([+-]?\d+(?:\.\d+)?)')
TIME_PATTERN = re.compile(
    r'Exhaustive evaluation wall time seconds:\s*(\d+)')


def tag(value):
    return str(value).replace('.', 'p')


def read_result(path):
    if not path.is_file():
        return None
    text = path.read_text(errors='replace')
    metrics = METRIC_PATTERN.findall(text)
    if not metrics:
        return None
    wall_times = TIME_PATTERN.findall(text)
    values = [float(value) for value in metrics[-1]]
    return {
        'task': values[0],
        'top1': values[1],
        'top5': values[2],
        'loss': values[3],
        'forgetting': values[4],
        'backward': values[5],
        'seconds': int(wall_times[-1]) if wall_times else None,
    }


def main():
    if len(sys.argv) < 5:
        raise SystemExit(
            'Usage: summarize_exhaustive_ablation.py '
            'OUTPUT_ROOT RUN_BASENAME TEMPERATURE PRIOR [PRIOR ...]')
    output_root = Path(sys.argv[1])
    run_basename = sys.argv[2]
    temperature = sys.argv[3]
    priors = sys.argv[4:]
    rows = []
    for prior in priors:
        log_path = output_root / (
            f'{run_basename}_eval_exhaustive_p{tag(prior)}_'
            f't{tag(temperature)}.log')
        result = read_result(log_path)
        if result is not None:
            rows.append((prior, result))

    print('# Exhaustive rematching prior ablation')
    print()
    print(f'Logit temperature: `{temperature}`')
    print()
    print('| TII prior | Acc@task | Acc@1 | Acc@5 | Loss | Forgetting | Backward | Wall time |')
    print('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for prior, result in rows:
        seconds = '-'
        if result['seconds'] is not None:
            seconds = f"{result['seconds'] / 60.0:.1f} min"
        print(
            f"| {prior} | {result['task']:.4f} | {result['top1']:.4f} | "
            f"{result['top5']:.4f} | {result['loss']:.4f} | "
            f"{result['forgetting']:.4f} | {result['backward']:.4f} | {seconds} |")
    if rows:
        best_prior, best_result = max(
            rows,
            key=lambda item: (
                item[1]['top1'], -item[1]['forgetting'], item[1]['task']))
        print()
        print(
            f"Best by Acc@1 (then lower forgetting): prior `{best_prior}`, "
            f"Acc@1 `{best_result['top1']:.4f}`, "
            f"Forgetting `{best_result['forgetting']:.4f}`.")


if __name__ == '__main__':
    main()
