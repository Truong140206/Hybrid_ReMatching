from training_scripts.compare_imagenet_r_taskwise import (
    build_report,
    parse_accuracy_matrix,
    task_retention,
)


def make_log(path, values):
    lines = []
    for stage, tasks in values.items():
        for task, acc1 in tasks.items():
            lines.extend([
                f'Test: [Task {task}] Total time: 0:00:01',
                f'- Acc@task {acc1 + 1:.3f} Acc@1 {acc1:.3f} '
                f'Acc@5 {acc1 + 5:.3f} loss 1.000',
            ])
        lines.append(
            f'[Average accuracy till task{stage}] Acc@1: 0.0000')
    path.write_text('\n'.join(lines), encoding='utf-8')


def test_taskwise_parser_reconstructs_peak_final_and_backward(tmp_path):
    baseline_path = tmp_path / 'baseline.log'
    proposal_path = tmp_path / 'proposal.log'
    baseline = {}
    proposal = {}
    for stage in range(1, 11):
        baseline[stage] = {
            task: 80.0 + task - max(0, stage - task)
            for task in range(1, stage + 1)
        }
        proposal[stage] = {
            task: 81.0 + task - 1.5 * max(0, stage - task)
            for task in range(1, stage + 1)
        }
    make_log(baseline_path, baseline)
    make_log(proposal_path, proposal)

    baseline_matrix = parse_accuracy_matrix(baseline_path)
    proposal_matrix = parse_accuracy_matrix(proposal_path)
    retention = task_retention(proposal_matrix)
    report = build_report(baseline_matrix, proposal_matrix)

    assert retention[1]['initial'] == 82.0
    assert retention[1]['final'] == 68.5
    assert retention[1]['forgetting'] == 13.5
    assert retention[1]['backward'] == -13.5
    assert 'Aggregate old-task diagnosis' in report
    assert 'Task 1:' in report

def test_taskwise_parser_accepts_zero_based_progress_rows(tmp_path):
    path = tmp_path / 'zero_based.log'
    lines = []
    for stage in range(1, 11):
        for task in range(stage):
            acc1 = 70.0 + task
            lines.append(
                f'Test: [Task {task}]  [41/42]  Loss: 1.0 (0.9000)  '
                f'Acc@1: 75.0 ({acc1:.4f})  Acc@5: 95.0 (94.0000)  '
                f'Acc@task: 80.0 (79.0000)')
        lines.append(
            f'[Average accuracy till task{stage}] Acc@1: 0.0000')
    path.write_text('\n'.join(lines), encoding='utf-8')

    matrix = parse_accuracy_matrix(path)
    assert matrix[10][1]['Acc@1'] == 70.0
    assert matrix[10][10]['Acc@1'] == 79.0
    assert matrix[10][1]['Loss'] == 0.9
