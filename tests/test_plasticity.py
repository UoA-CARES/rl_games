import os
import tempfile
import warnings
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

import torch
from torch import nn

from rl_games.algos_torch.plasticity import (
    _STATE_NESTED_TYPES,
    _STATE_SKIP_FIELDS,
    NetworkPlasticityManager,
    SiteState,
)

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


# =============================================================================
# Checkpoint save/load
# =============================================================================


def _ckpt_manager(hidden=(8,), device="cpu", **kwargs):
    """A manager over a trunk with the given hidden widths, on `device`."""
    torch.manual_seed(0)
    layers, in_features = [], 4
    for width in hidden:
        layers += [nn.Linear(in_features, width), nn.ReLU()]
        in_features = width
    layers.append(nn.Linear(in_features, 2))
    model = nn.Sequential(*layers).to(device)
    with torch.no_grad():
        for layer in model:
            if isinstance(layer, nn.Linear):
                layer.bias.add_(1.0)  # keep units firing, as in _tiny_manager
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    kwargs.setdefault("log_interval", 1)
    kwargs.setdefault("rank_interval", NEVER)
    kwargs.setdefault("name", "demo")
    manager = NetworkPlasticityManager(model, optimizer, **kwargs)
    return model, optimizer, manager


def _capture_steps(model, optimizer, manager, steps=3, device="cpu"):
    """Drive both capture paths so every site_state field is non-trivial."""
    for _ in range(steps):
        with manager.capture_training():
            model(torch.randn(16, 4, device=device)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        with manager.capture_metrics():
            model(torch.randn(16, 4, device=device))


def _iter_state_tensors(obj, path=""):
    if is_dataclass(obj):
        for field in fields(obj):
            yield from _iter_state_tensors(getattr(obj, field.name), path + "." + field.name)
    elif torch.is_tensor(obj):
        yield path, obj


def _walk_tensors(node):
    if torch.is_tensor(node):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk_tensors(value)


def _populated_state_dict(**kwargs):
    model, optimizer, manager = _ckpt_manager(**kwargs)
    _capture_steps(model, optimizer, manager)
    manager.step_count = 42
    return manager, manager.state_dict()


def test_state_dict_round_trip_restores_stats():
    source, saved = _populated_state_dict()
    _, _, target = _ckpt_manager()

    assert target.load_state_dict(saved, strict=True) is True
    assert target.step_count == source.step_count
    assert set(target.site_states) == set(source.site_states)

    compared = 0
    for name, restored in target.site_states.items():
        original = dict(_iter_state_tensors(source.site_states[name]))
        for path, tensor in _iter_state_tensors(restored):
            assert torch.equal(tensor, original[path]), path
            compared += 1
    # 14 per site: utility 4, activity window/lifetime 2+2, redo 2,
    # knife window/lifetime 2+2. Walked off dataclasses.fields, so a new
    # field is picked up here automatically.
    assert compared == 14 * len(target.site_states) > 0


def test_state_dict_is_a_snapshot_not_a_view():
    # On a CPU model .cpu() is a no-op returning the same object, so without an
    # explicit copy the "snapshot" would track the live tensors the hooks keep
    # mutating in place.
    model, optimizer, manager = _ckpt_manager()
    _capture_steps(model, optimizer, manager)

    saved = manager.state_dict()
    before = {p: t.clone() for p, t in _iter_state_tensors(manager.site_states["0"])}

    _capture_steps(model, optimizer, manager, steps=3)
    assert not torch.equal(manager.site_states["0"].utility.age, before[".utility.age"])

    for path, tensor in before.items():
        keys = [k for k in path.split(".") if k]
        node = saved["site_states"]["0"]
        for key in keys:
            node = node[key]
        assert torch.equal(node, tensor), path


def test_state_dict_drops_behaviour_window():
    model, optimizer, manager = _ckpt_manager()
    _capture_steps(model, optimizer, manager)
    assert manager.site_states["0"].activity.behaviour_rows > 0

    saved = manager.state_dict()
    assert "behaviour_chunks" not in saved["site_states"]["0"]["activity"]
    assert "behaviour_rows" not in saved["site_states"]["0"]["activity"]

    _, _, target = _ckpt_manager()
    assert target.load_state_dict(saved, strict=True) is True
    assert target.site_states["0"].activity.behaviour_chunks == []
    assert target.site_states["0"].activity.behaviour_rows == 0


def test_state_dict_tensors_are_on_cpu():
    _, saved = _populated_state_dict()
    for tensor in _walk_tensors(saved["site_states"]):
        assert tensor.device.type == "cpu"


def test_load_places_tensors_on_producer_device():
    _, saved = _populated_state_dict()
    model, _, target = _ckpt_manager()
    assert target.load_state_dict(saved, strict=True) is True

    expected = model[0].weight.device
    for path, tensor in _iter_state_tensors(target.site_states["0"]):
        assert tensor.device == expected, path
        assert tensor.dtype is torch.float32, path


def test_load_places_tensors_on_cuda():
    # Plain guard rather than a pytest marker: the __main__ runner below calls
    # each test directly and would not understand a skip.
    if not torch.cuda.is_available():
        return

    _, saved = _populated_state_dict()  # CPU tensors, as state_dict always makes
    model, optimizer, target = _ckpt_manager(device="cuda")
    assert target.load_state_dict(saved, strict=True) is True

    for path, tensor in _iter_state_tensors(target.site_states["0"]):
        assert tensor.is_cuda, path

    # The restored tensors have to be usable by the hooks' in-place updates,
    # which is where a device mismatch would actually surface.
    _capture_steps(model, optimizer, target, steps=1, device="cuda")
    assert target.summary(force=True) != {}


def test_load_warns_and_skips_on_site_name_mismatch():
    _, saved = _populated_state_dict(hidden=(8, 6))  # two sites
    _, _, target = _ckpt_manager(hidden=(8,))        # one site

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert target.load_state_dict(saved) is False
    assert "topology changed" in str(caught[0].message)

    # Nothing half-applied.
    assert target.site_states == {}
    assert target.step_count == 0


def test_load_warns_and_skips_on_width_mismatch():
    _, saved = _populated_state_dict(hidden=(8,))
    _, _, target = _ckpt_manager(hidden=(16,))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert target.load_state_dict(saved) is False
    assert "width(s) changed" in str(caught[0].message)
    assert target.site_states == {}
    assert target.step_count == 0


def test_load_warns_and_skips_on_version_mismatch():
    _, saved = _populated_state_dict()
    saved["version"] += 1
    _, _, target = _ckpt_manager()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert target.load_state_dict(saved) is False
    assert "version" in str(caught[0].message)
    assert target.site_states == {}


def test_load_strict_raises_instead_of_warning():
    _, saved = _populated_state_dict(hidden=(8,))
    _, _, target = _ckpt_manager(hidden=(16,))
    try:
        target.load_state_dict(saved, strict=True)
    except ValueError:
        pass
    else:
        assert False, "strict=True should raise on a mismatch"


def test_load_warns_but_still_loads_on_config_change():
    _, saved = _populated_state_dict(utility_decay=0.99)
    _, _, target = _ckpt_manager(utility_decay=0.9)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert target.load_state_dict(saved) is True
    assert "accumulation settings changed" in str(caught[0].message)
    assert target.site_states != {}


def test_load_into_empty_site_states():
    # site_states is populated lazily by the hooks, so at restore time - which
    # runs right after init_plasticity(), before any forward pass - it is empty.
    # The load cannot lean on _ensure_site_state having run.
    _, saved = _populated_state_dict()
    _, _, target = _ckpt_manager()
    assert target.site_states == {}
    assert target.load_state_dict(saved, strict=True) is True
    assert target.site_states != {}


def test_load_tolerates_empty_saved_site_states():
    # The shape of every real checkpoint today: the capture contexts are not
    # wired into the training loop, so site_order is full and site_states is
    # empty. This must be silent, not a warning on every resume.
    _, _, source = _ckpt_manager()
    source.step_count = 7
    saved = source.state_dict()
    assert saved["site_order"] == ["0"]
    assert saved["site_states"] == {}

    _, _, target = _ckpt_manager()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert target.load_state_dict(saved, strict=True) is True
    assert caught == []
    assert target.step_count == 7
    assert target.site_states == {}


def test_cadence_resumes_from_restored_step_count():
    _, _, source = _ckpt_manager(log_interval=10)
    source.step_count = 9
    saved = source.state_dict()

    _, _, resumed = _ckpt_manager(log_interval=10)
    assert resumed.load_state_dict(saved, strict=True) is True

    _, _, control = _ckpt_manager(log_interval=10)

    assert resumed.summary() != {}  # step_count 9 -> 10, a logging boundary
    assert control.summary() == {}  # step_count 0 -> 1


def test_hooks_survive_model_load_state_dict():
    # The executable form of why the manager needs no model/optimizer
    # re-pointing after a restore: load_state_dict copies in place, so the
    # module and Parameter objects the hooks are bound to keep their identity.
    model, optimizer, manager = _ckpt_manager()
    _capture_steps(model, optimizer, manager)

    site = manager.sites[0]
    weight_id = id(site.producer_module.weight)
    module_id = id(site.hook_module)
    age_before = manager.site_states["0"].utility.age.clone()

    other, _, _ = _ckpt_manager()
    with torch.no_grad():
        for param in other.parameters():
            param.add_(1.0)
    model.load_state_dict(other.state_dict())  # default assign=False

    assert id(site.producer_module.weight) == weight_id
    assert id(site.hook_module) == module_id

    _capture_steps(model, optimizer, manager, steps=1)
    assert (manager.site_states["0"].utility.age > age_before).all()


def test_agent_restore_helper_tolerates_all_shapes():
    from rl_games.common.a2c_common import A2CBase

    _, saved = _populated_state_dict(name="shared")

    # No managers: with and without saved state in the checkpoint.
    for payload in (None, {}, {"shared": saved}):
        A2CBase._restore_plasticity_state(SimpleNamespace(plasticity_managers=[]), payload)

    # Managers present, every checkpoint shape.
    for payload in (None, {}, {"shared": saved}, {"unknown": saved}):
        _, _, manager = _ckpt_manager(name="shared")
        A2CBase._restore_plasticity_state(
            SimpleNamespace(plasticity_managers=[manager]), payload
        )


def test_agent_restore_helper_applies_state():
    from rl_games.common.a2c_common import A2CBase

    source, saved = _populated_state_dict(name="shared")
    _, _, manager = _ckpt_manager(name="shared")

    A2CBase._restore_plasticity_state(
        SimpleNamespace(plasticity_managers=[manager]), {"shared": saved}
    )
    assert manager.step_count == source.step_count
    assert manager.site_states != {}


def test_torch_save_load_round_trip():
    # Proves nothing unpicklable (hook handles, live module refs, dataclass
    # instances) leaked into the payload, and that it survives torch.load's
    # weights_only=True default from torch 2.6 on.
    _, saved = _populated_state_dict()

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "plasticity.pth")
        torch.save(saved, path)
        reloaded = torch.load(path)

    _, _, target = _ckpt_manager()
    assert target.load_state_dict(reloaded, strict=True) is True
    assert target.step_count == saved["step_count"]


def test_state_tree_field_coverage():
    # Drift guard: a field added to any state dataclass must end up skipped,
    # nested, or dumped as a tensor - never silently dropped.
    _, saved = _populated_state_dict()

    def check(cls, dumped):
        names = {field.name for field in fields(cls)}
        assert names <= set(dumped) | set(_STATE_SKIP_FIELDS), cls.__name__
        for name, value in dumped.items():
            nested = _STATE_NESTED_TYPES.get((cls, name))
            if nested is not None:
                assert isinstance(value, dict), (cls.__name__, name)
                check(nested, value)
            else:
                assert torch.is_tensor(value), (cls.__name__, name)

    for dumped in saved["site_states"].values():
        check(SiteState, dumped)

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
