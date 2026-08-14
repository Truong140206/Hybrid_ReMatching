from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tii_config_exposes_pilot_and_protocol_flags():
    source = (ROOT / 'configs' / 'imr_hideprompt_5e.py').read_text()
    assert "--max_train_tasks" in source
    assert "--strict_exemplar_free" in source


def test_tii_engine_limits_training_without_changing_dataset_split():
    source = (ROOT / 'engines' / 'hide_tii_engine.py').read_text()
    assert "task_count = args.num_tasks" in source
    assert "task_count = min(task_count, max_train_tasks)" in source
    assert "for task_id in range(task_count):" in source


def test_tii_cfs_pilot_is_a_single_variable_ablation():
    script = (
        ROOT / 'training_scripts' / 'run_imagenet_r_tii_cfs_4090.sh'
    ).read_text()
    assert "--ca_storage_efficient_method covariance" in script
    assert "--strict_exemplar_free" in script
    assert "--cfs_sampling" in script
    assert "--cfs_paper_style" in script
    assert "--cfs_epochs 200" in script
    assert 'CFS_SELECTION_RATIO="${CFS_SELECTION_RATIO:-0.5}"' in script
    assert '--cfs_selection_ratio "${CFS_SELECTION_RATIO}"' in script
    assert "--semantic_" not in script
    assert "--cfs_boundary_replay" not in script
    assert "--crct_use_all_samples" not in script
    assert "--crct_balanced_batches" not in script
    assert '--max_train_tasks "${TASK_COUNT}"' in script


def test_tii_cfs_pilot_uses_strict_five_metric_gate():
    script = (
        ROOT / 'training_scripts' / 'run_imagenet_r_tii_cfs_4090.sh'
    ).read_text()
    for metric in ('Acc@1', 'Acc@5', 'Loss', 'Forgetting', 'Backward'):
        assert metric in script
    assert "PILOT_GATE=" in script


def test_tii_evaluation_reports_operational_task_accuracy():
    engine = (ROOT / 'engines' / 'hide_tii_engine.py').read_text()
    trainer = (ROOT / 'trainers' / 'tii_trainer.py').read_text()
    assert "target_task_map[value.item()] for value in predicted_class" in engine
    assert "Acc@task" in engine
    assert "for task_id in range(task_count):" in trainer


def test_task_accuracy_eval_reuses_checkpoints_without_training():
    script = (
        ROOT / 'training_scripts' /
        'eval_imagenet_r_tii_task_accuracy_4090.sh'
    ).read_text()
    assert "--eval" in script
    assert "--max_train_tasks \"${TASK_COUNT}\"" in script
    assert "TII_ROUTING_GATE=" in script
    assert "STRICT_ALL_METRIC_GATE=" in script
    assert "--cfs_sampling" not in script
