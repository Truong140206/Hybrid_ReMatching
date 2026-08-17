import torch

from engines.progressive_oracle_audit import router_recall_diagnostics


def test_router_recall_ranks_winner_second():
    # task maxima: task0=5, task1=3, task2=4 -> winner task2 sits at rank 2.
    tii_logits = torch.tensor([[5.0, 1.0, 3.0, 0.0, 4.0, 2.0]])
    class_mask = [[0, 1], [2, 3], [4, 5]]
    winner = torch.tensor([2])
    diag = router_recall_diagnostics(tii_logits, class_mask, 3, winner)

    assert diag['router_max_mean_rank'].item() == 2.0
    assert not bool(diag['router_max_recall_1'][0])
    assert bool(diag['router_max_recall_2'][0])
    assert bool(diag['router_max_recall_3'][0])
    # energy (logsumexp) preserves this ordering.
    assert diag['router_energy_mean_rank'].item() == 2.0
    assert bool(diag['router_energy_recall_2'][0])


def test_router_recall_winner_is_top1():
    tii_logits = torch.tensor([[9.0, 8.0, 1.0, 0.0, 2.0, 3.0]])
    class_mask = [[0, 1], [2, 3], [4, 5]]
    winner = torch.tensor([0])
    diag = router_recall_diagnostics(tii_logits, class_mask, 3, winner)

    assert diag['router_max_mean_rank'].item() == 1.0
    assert bool(diag['router_max_recall_1'][0])
    assert bool(diag['router_max_recall_2'][0])


def test_router_recall_keys_present_for_all_routers():
    tii_logits = torch.randn(4, 6)
    class_mask = [[0, 1], [2, 3], [4, 5]]
    winner = torch.tensor([0, 1, 2, 1])
    diag = router_recall_diagnostics(tii_logits, class_mask, 3, winner)

    for name in ('max', 'energy', 'margin', 'mean'):
        assert diag['router_{}_mean_rank'.format(name)].shape == (4,)
        for k in (1, 2, 3, 4):
            key = 'router_{}_recall_{}'.format(name, k)
            assert diag[key].shape == (4,)
            assert diag[key].dtype == torch.bool
