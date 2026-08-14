from training_scripts.summarize_imagenet_r_multiseed import (
    build_markdown,
    collect_runs,
)


ROWS = {
    'Baseline': ('Acc@task: 77.0000\tAcc@1: 73.0000\tAcc@5: 86.0000\t'
                 'Loss: 1.2000\tForgetting: 3.3000\tBackward: -3.0000'),
    'Exhaustive': ('Acc@task: 80.0000\tAcc@1: 75.0000\tAcc@5: 88.0000\t'
                   'Loss: 1.1000\tForgetting: 2.9000\tBackward: -2.8000'),
    'Proposal': ('Acc@task: 79.0000\tAcc@1: 74.0000\tAcc@5: 87.0000\t'
                 'Loss: 1.1500\tLoRA/sample: 5.0000\tForwardCalls/sample: 2.0000\t'
                 'Forgetting: 3.0000\tBackward: -2.9000'),
}


def write_seed(root, seed, proposal_acc1=74.0):
    name = f'imr_lora_rank8_baseline_10tasks_seed{seed}'
    paths = {
        'Baseline': root / f'{name}_eval_conventional.log',
        'Exhaustive': root / f'{name}_eval_vectorized_exhaustive_c4_p0p3_t1p0.log',
        'Proposal': root / f'{name}_eval_prediction_proposal_i2_p3_c5_tiicomplete_strict.log',
    }
    for method, path in paths.items():
        row = ROWS[method]
        if method == 'Proposal':
            row = row.replace('Acc@1: 74.0000', f'Acc@1: {proposal_acc1:.4f}')
        suffix = ''
        if method == 'Baseline':
            suffix = '\nConventional evaluation wall time seconds: 250'
        elif method == 'Exhaustive':
            suffix = '\nVectorized exhaustive wall time seconds: 500'
        elif method == 'Proposal':
            suffix = '\nPrediction-proposal evaluation wall time seconds: 300'
        path.write_text(
            f'[Average accuracy till task10]\t{row}{suffix}\n',
            encoding='utf-8',
        )


def test_multiseed_summary_reports_mean_std_and_per_seed_gate(tmp_path):
    write_seed(tmp_path, 42, proposal_acc1=74.0)
    write_seed(tmp_path, 43, proposal_acc1=74.2)
    records = collect_runs(
        tmp_path, [42, 43], 'imr_lora_rank8_baseline_10tasks_seed{seed}')
    report = build_markdown(records, [42, 43])

    assert '74.1000 ± 0.1414' in report
    assert '300.0 ± 0.0' in report
    assert report.count('| PASS |') == 2


def test_multiseed_summary_fails_when_a_required_log_is_missing(tmp_path):
    import pytest

    write_seed(tmp_path, 42)
    (tmp_path / 'imr_lora_rank8_baseline_10tasks_seed42_eval_conventional.log').unlink()
    with pytest.raises(FileNotFoundError, match='Missing required logs'):
        collect_runs(
            tmp_path, [42], 'imr_lora_rank8_baseline_10tasks_seed{seed}')
