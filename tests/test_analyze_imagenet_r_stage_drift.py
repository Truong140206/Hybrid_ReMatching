from training_scripts.analyze_imagenet_r_stage_drift import (
    build_stage_drift_report,
    parse_stage_drift_matrix,
)
from training_scripts.compare_imagenet_r_taskwise import parse_accuracy_matrix


def _write_stage_drift_log(path):
    lines = []
    for stage in range(1, 11):
        for task in range(1, stage + 1):
            age = stage - task
            local = 90.0 - 0.5 * age
            seen = local - 0.25 * age
            proposal = seen - 0.1 * age
            lines.extend([
                f'Test: [Task {task}] Total time: 0:00:01',
                f'- Acc@task {proposal:.3f} Acc@1 {proposal:.3f} '
                f'Acc@5 {proposal + 5:.3f} loss {1 + age / 10:.3f}',
                f'* StageDriftAudit OwnLocalAcc@1 {local:.3f} '
                f'OwnSeenAcc@1 {seen:.3f} '
                f'OwnLocalLoss {0.5 + age / 100:.4f} '
                f'OwnSeenLoss {0.5 + age / 50:.4f} '
                f'OwnSeenTaskAcc {seen + 2:.3f} '
                f'LocalToSeenFailure {0.25 * age:.3f}',
            ])
        lines.append(
            f'[Average accuracy till task{stage}] Acc@1: 0.0000')
    path.write_text('\n'.join(lines), encoding='utf-8')


def test_stage_drift_parser_and_report(tmp_path):
    log_path = tmp_path / 'stage_drift.log'
    _write_stage_drift_log(log_path)

    drift = parse_stage_drift_matrix(log_path)
    proposal = parse_accuracy_matrix(log_path)
    report = build_stage_drift_report(drift, proposal)

    assert drift[10][1]['OwnLocalAcc@1'] == 85.5
    assert drift[10][1]['OwnSeenAcc@1'] == 83.25
    assert 'Dominant measured signal: TRUE_ADAPTER_LOCAL_DRIFT' in report
    assert 'Focus tasks 6-8' in report
    assert 'CROSS_TASK_SCORE_COMPETITION' in report