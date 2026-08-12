from types import SimpleNamespace

import torch
from torch import nn

from engines.hrm_lora_wtp_and_tap_engine import (
    _macro_crct_metrics,
    _select_crct_validation_alpha,
)


class TinyClassifier(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.fc_norm = nn.Identity()
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(weight, dtype=torch.float32))


def _args():
    return SimpleNamespace(
        crct_validation_steps=10,
        crct_validation_max_old_acc_drop=0.0,
        crct_validation_min_acc_gain=0.0,
        crct_validation_min_ce_gain=0.0,
    )


def test_old_class_metrics_ignore_new_class_targets():
    logits = torch.tensor([
        [4.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 3.0]])
    targets = torch.tensor([0, 0, 1, 1])

    metrics = _macro_crct_metrics(logits, targets, class_ids=[0])

    assert metrics['accuracy'] == 100.0
    assert set(metrics['per_class_accuracy']) == {0}


def test_gate_accepts_classifier_that_improves_all_classes():
    teacher = TinyClassifier([[0.2, 0.0], [0.0, 0.2]])
    student = TinyClassifier([[2.0, -1.0], [-1.0, 2.0]])
    anchors = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.8]])
    targets = torch.tensor([0, 0, 1, 1])

    alpha, metrics = _select_crct_validation_alpha(
        student,
        teacher.fc_norm,
        teacher.head,
        anchors,
        targets,
        seen_classes=[0, 1],
        old_classes=[0],
        args=_args(),
    )

    assert alpha == 1.0
    assert metrics['selected_all']['ce'] < metrics['teacher_all']['ce']


def test_gate_rolls_back_classifier_that_hurts_old_class():
    teacher = TinyClassifier([[2.0, -1.0], [-1.0, 2.0]])
    student = TinyClassifier([[-2.0, 1.0], [2.0, -1.0]])
    anchors = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.8]])
    targets = torch.tensor([0, 0, 1, 1])

    alpha, metrics = _select_crct_validation_alpha(
        student,
        teacher.fc_norm,
        teacher.head,
        anchors,
        targets,
        seen_classes=[0, 1],
        old_classes=[0],
        args=_args(),
    )

    assert alpha == 0.0
    assert metrics['selected_old']['accuracy'] == metrics['teacher_old']['accuracy']
