import torch
from torch import nn

from rl_games.algos_torch.plasticity import NetworkPlasticityManager

# A "never" interval. 0 would be the natural way to say it, but
# _summary_schedule divides by these unguarded (plasticity.py:674,680), so 0
# raises ZeroDivisionError - see test_schedule_zero_interval_currently_raises.
NEVER = 10 ** 9


def _schedule_manager(**kwargs):
    # enabled=False skips site discovery/hook registration entirely, so
    # _summary_schedule can be tested as pure cadence arithmetic without a
    # real model or optimizer.
    return NetworkPlasticityManager(
        model=nn.Identity(), optimizer=None, enabled=False, **kwargs
    )


def _schedule_at(manager, step_count, force=False):
    # _summary_schedule reads self.step_count but never advances it - only
    # summary() does, exactly once per call. Setting it directly lets each
    # cadence case be written as "what happens at step N".
    manager.step_count = step_count
    return manager._summary_schedule(force)


def test_schedule_no_log_before_first_boundary():
    m = _schedule_manager(log_interval=100, rank_interval=NEVER, knife_interval=1)
    should_log, should_rank, should_knife = _schedule_at(m, 50)
    assert should_log is False


def test_schedule_logs_on_exact_boundary():
    m = _schedule_manager(log_interval=100, rank_interval=NEVER, knife_interval=1)
    should_log, _, _ = _schedule_at(m, 100)
    assert should_log is True


def test_schedule_force_always_logs():
    m = _schedule_manager(log_interval=100, rank_interval=NEVER, knife_interval=1)
    should_log, _, _ = _schedule_at(m, 1, force=True)
    assert should_log is True


def test_schedule_zero_interval_currently_raises():
    # Documents a real bug rather than the intended behaviour: log_interval and
    # rank_interval are user-settable YAML keys (a2c_common.py:75-77) with no
    # validation, and _summary_schedule divides by both unguarded
    # (plasticity.py:674 and :680), so `plasticity: {log_interval: 0}` takes
    # down the first write_stats. Fixing that is out of scope here; when it is
    # fixed (a `<= 0 -> never` guard, or validation in
    # _plasticity_manager_kwargs) this test should be rewritten to assert the
    # new behaviour.
    m = _schedule_manager(log_interval=0, rank_interval=NEVER, knife_interval=1)
    try:
        _schedule_at(m, 10_000_000)
    except ZeroDivisionError:
        pass
    else:
        assert False, 'expected ZeroDivisionError from log_interval=0'

    m2 = _schedule_manager(log_interval=100, rank_interval=0, knife_interval=1)
    try:
        _schedule_at(m2, 10_000_000)
    except ZeroDivisionError:
        pass
    else:
        assert False, 'expected ZeroDivisionError from rank_interval=0'


def test_schedule_rank_cadence_independent_of_log_cadence():
    m = _schedule_manager(log_interval=100, rank_interval=1000, knife_interval=1)
    should_log, should_rank, _ = _schedule_at(m, 200)
    assert should_log is True    # a multiple of 100
    assert should_rank is False  # not a multiple of 1000

    should_log2, should_rank2, _ = _schedule_at(m, 1000)
    assert should_log2 is True
    assert should_rank2 is True


def test_schedule_compute_rank_false_never_ranks():
    m = _schedule_manager(log_interval=100, rank_interval=100, compute_rank=False)
    _, should_rank, _ = _schedule_at(m, 100)
    assert should_rank is False


def test_schedule_knife_interval_gates_every_nth_log():
    # knife_interval=2 -> should_knife fires on the 2nd, 4th... log event
    # (log_count % 2 == 0), not every log event.
    m = _schedule_manager(log_interval=100, rank_interval=NEVER, knife_interval=2)

    _, _, should_knife_1 = _schedule_at(m, 100)  # log_count=1
    assert should_knife_1 is False

    _, _, should_knife_2 = _schedule_at(m, 200)  # log_count=2
    assert should_knife_2 is True

    _, _, should_knife_3 = _schedule_at(m, 300)  # log_count=3
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


def test_summary_advances_exactly_one_step_per_call():
    # The cadence is plain modulo on step_count (plasticity.py:669-671), which
    # is only safe because step_count can never jump. This is that guarantee:
    # summary() advances it by exactly 1 per call, so no multiple of
    # log_interval can ever be stepped over. (An enabled manager is required -
    # summary() returns early on a disabled one, before the increment.)
    _, _, manager = _tiny_manager(log_interval=100, rank_interval=NEVER)
    assert manager.step_count == 0
    for expected in range(1, 6):
        manager.summary()
        assert manager.step_count == expected


def test_summary_empty_before_threshold_crossed():
    model, optimizer, manager = _tiny_manager(log_interval=1000, rank_interval=NEVER)
    _run_training_step(model, optimizer, manager)
    info = manager.summary()  # step_count 0 -> 1
    assert info == {}


def test_summary_populated_after_threshold_crossed():
    model, optimizer, manager = _tiny_manager(log_interval=1000, rank_interval=NEVER)
    _run_training_step(model, optimizer, manager)
    manager.step_count = 999
    info = manager.summary()  # step_count 999 -> 1000
    assert info != {}
    assert "demo/dead_units_lifetime_frac" in info


def test_summary_disabled_manager_always_empty():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    manager = NetworkPlasticityManager(model, optimizer, enabled=False)
    assert manager.summary(force=True) == {}


def test_summary_resets_window_but_keeps_lifetime():
    model, optimizer, manager = _tiny_manager(log_interval=100, rank_interval=NEVER)
    _run_training_step(model, optimizer, manager)
    manager.step_count = 99
    manager.summary()  # step_count 99 -> 100

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
