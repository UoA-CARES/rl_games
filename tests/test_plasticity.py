import torch
from torch import nn

from rl_games.algos_torch.plasticity import NetworkPlasticityManager


def _schedule_manager(**kwargs):
    # enabled=False skips site discovery/hook registration entirely, so
    # _summary_schedule can be tested as pure cadence arithmetic without a
    # real model or optimizer.
    return NetworkPlasticityManager(
        model=nn.Identity(), optimizer=None, enabled=False, **kwargs
    )


def test_schedule_no_log_before_first_boundary():
    m = _schedule_manager(log_interval_frames=100, rank_interval_frames=0, knife_interval=1)
    should_log, should_rank, should_knife = m._summary_schedule(frame=50, force=False)
    assert should_log is False
    assert m._last_logged_frame == 0


def test_schedule_logs_on_exact_boundary():
    m = _schedule_manager(log_interval_frames=100, rank_interval_frames=0, knife_interval=1)
    should_log, _, _ = m._summary_schedule(frame=100, force=False)
    assert should_log is True
    assert m._last_logged_frame == 100


def test_schedule_logs_on_irregular_crossing():
    # frame jumps straight from 80 to 250 in one call (num_envs * horizon
    # steps rarely land on a round number) - two boundaries (100, 200) are
    # skipped over, but the crossing must still be detected.
    m = _schedule_manager(log_interval_frames=100, rank_interval_frames=0, knife_interval=1)
    m._last_logged_frame = 80
    should_log, _, _ = m._summary_schedule(frame=250, force=False)
    assert should_log is True


def test_schedule_no_crossing_within_same_bucket():
    m = _schedule_manager(log_interval_frames=100, rank_interval_frames=0, knife_interval=1)
    m._last_logged_frame = 80
    should_log, _, _ = m._summary_schedule(frame=95, force=False)
    assert should_log is False
    assert m._last_logged_frame == 80  # unchanged - no log happened


def test_schedule_force_always_logs():
    m = _schedule_manager(log_interval_frames=100, rank_interval_frames=0, knife_interval=1)
    should_log, _, _ = m._summary_schedule(frame=1, force=True)
    assert should_log is True


def test_schedule_zero_interval_disables_periodic_log():
    m = _schedule_manager(log_interval_frames=0, rank_interval_frames=0, knife_interval=1)
    should_log, _, _ = m._summary_schedule(frame=10_000_000, force=False)
    assert should_log is False
    should_log_forced, _, _ = m._summary_schedule(frame=10_000_000, force=True)
    assert should_log_forced is True


def test_schedule_rank_cadence_independent_of_log_cadence():
    m = _schedule_manager(
        log_interval_frames=100, rank_interval_frames=1000, knife_interval=1
    )
    should_log, should_rank, _ = m._summary_schedule(frame=200, force=False)
    assert should_log is True   # crossed a 100-frame boundary
    assert should_rank is False  # hasn't crossed a 1000-frame boundary yet

    should_log2, should_rank2, _ = m._summary_schedule(frame=1000, force=False)
    assert should_log2 is True
    assert should_rank2 is True


def test_schedule_compute_rank_false_never_ranks():
    m = _schedule_manager(
        log_interval_frames=100, rank_interval_frames=100, compute_rank=False
    )
    _, should_rank, _ = m._summary_schedule(frame=100, force=False)
    assert should_rank is False


def test_schedule_knife_interval_gates_every_nth_log():
    # knife_interval=2 -> should_knife fires on the 1st, 3rd, 5th... log
    # event (log_count % 2 == 0), not every log event.
    m = _schedule_manager(log_interval_frames=100, rank_interval_frames=0, knife_interval=2)

    _, _, should_knife_1 = m._summary_schedule(frame=100, force=False)  # log_count=1
    assert should_knife_1 is False

    _, _, should_knife_2 = m._summary_schedule(frame=200, force=False)  # log_count=2
    assert should_knife_2 is True

    _, _, should_knife_3 = m._summary_schedule(frame=300, force=False)  # log_count=3
    assert should_knife_3 is False


def _tiny_manager(**kwargs):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8, bias=True), nn.ReLU(), nn.Linear(8, 2))
    with torch.no_grad():
        # bias the hidden layer positive so units actually fire - otherwise
        # a random draw can produce an all-dead layer (as we saw earlier)
        # and every metric collapses to a degenerate zero/one value.
        model[0].bias.add_(1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = NetworkPlasticityManager(model, optimizer, name="demo", **kwargs)
    return model, optimizer, manager


def _run_training_step(model, optimizer, manager):
    with manager.capture_training():
        x = torch.randn(16, 4)
        loss = model(x).sum()
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_summary_empty_before_threshold_crossed():
    model, optimizer, manager = _tiny_manager(log_interval_frames=1000, rank_interval_frames=0)
    _run_training_step(model, optimizer, manager)
    info = manager.summary(frame=500, force=False)
    assert info == {}
    assert manager._last_logged_frame == 0


def test_summary_populated_after_threshold_crossed():
    model, optimizer, manager = _tiny_manager(log_interval_frames=1000, rank_interval_frames=0)
    _run_training_step(model, optimizer, manager)
    info = manager.summary(frame=1000, force=False)
    assert info != {}
    assert "demo/dead_units_lifetime_frac" in info
    assert manager._last_logged_frame == 1000


def test_summary_disabled_manager_always_empty():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = NetworkPlasticityManager(model, optimizer, enabled=False)
    info = manager.summary(frame=10_000_000, force=True)
    assert info == {}


def test_summary_resets_window_but_keeps_lifetime():
    model, optimizer, manager = _tiny_manager(log_interval_frames=100, rank_interval_frames=0)
    _run_training_step(model, optimizer, manager)
    manager.summary(frame=100, force=False)

    site_name = manager.sites[0].name
    state = manager.site_states[site_name]
    assert state.activity.window.count.sum().item() == 0  # reset
    assert state.activity.lifetime.count.sum().item() > 0  # preserved


if __name__ == "__main__":
    import sys

    failures = 0
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
