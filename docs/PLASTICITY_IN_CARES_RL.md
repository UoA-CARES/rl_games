# How Plasticity Is Implemented in `cares_reinforcement_learning`

This documents the **existing, working** plasticity implementation in that
repo — for reference only (per supervisor), when reproducing the same pattern
inside rl_games' PPO agent here in `ard-isaaclab-tasks`.

## The three pieces

| File (in `cares_reinforcement_learning`) | Role |
|---|---|
| `networks/plasticity.py` | `NetworkPlasticityManager` — diagnostics + unit replacement |
| `algorithm/plasticity_adam.py` | `PlasticityAdam` — Adam variant with resettable per-unit state |
| `algorithm/configurations.py` (`PlasticityConfig`, line 86) | config dataclass (pydantic) controlling both of the above |

`NetworkPlasticityManager` also uses `networks/activation_functions.py` (for
`caf.GoLU`, one of its supported activation types) and needs `PlasticityConfig`
+ its `SubscriptableClass` base.

## Where it's wired in — `algorithm/policy/PPO.py`

This is the reference integration. Two independent networks (actor, critic),
each gets its own optimizer and its own manager, built identically:

```python
# PPO.py:144-172
optim_cls = PlasticityAdam if config.plasticity_config.enabled else torch.optim.Adam

self.actor_net_optimiser = optim_cls(list(self.actor_net.parameters()) + [self.log_std], lr=config.actor_lr, **config.actor_lr_params)
self.critic_net_optimiser = optim_cls(self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params)

self.actor_plasticity = NetworkPlasticityManager(model=self.actor_net, optimizer=self.actor_net_optimiser, config=config.plasticity_config, name="actor")
self.critic_plasticity = NetworkPlasticityManager(model=self.critic_net, optimizer=self.critic_net_optimiser, config=config.plasticity_config, name="critic")
```

Note: **`PlasticityAdam` is only required by the manager when
`replacement_enabled=True`** — `NetworkPlasticityManager.__init__` raises a
`TypeError` if you ask for replacement without it (it needs to zero per-unit
optimizer state on reset). If only diagnostics are wanted (`enabled=True`,
`replacement_enabled=False`), plain `Adam` still works — but `PPO.py` doesn't
draw that distinction; it swaps to `PlasticityAdam` whenever `enabled` is
true, unconditionally.

### 3 places the manager gets called from the training loop

1. **`act()` (rollout / action selection), line 219–261** — wraps the
   forward pass in `capture_metrics()` (not `capture_training()`) so
   rank/activity diagnostics reflect real rollout behaviour, without
   updating gradient-side (KNIFE) stats. Skipped entirely during
   `evaluation=True` (uses `nullcontext()` instead).

2. **`update_critic_minibatch()`, line 276–300** — forward pass, loss, and
   `.backward()` all happen inside `critic_plasticity.capture_training()`
   (line 282). `step_replacement()` is called *after* `optimiser.step()`
   (line 295), never before — so a freshly-reset unit's optimizer state
   isn't clobbered by a stale gradient computed pre-reset.

3. **`update_actor_minibatch()`, line 332–381** — identical pattern:
   `capture_training()` wraps forward+backward (line 341), `step_replacement()`
   called right after `optimiser.step()` (line 381).

Diagnostics are pulled out for logging separately, not inside the update
loop: `info.update(self.actor_plasticity.summary(prefix="actor"))` /
`...critic_plasticity.summary(prefix="critic")` (line 649–650).

## How `NetworkPlasticityManager` itself works

### Capture modes (why forward/backward must be wrapped explicitly)

The manager registers PyTorch hooks once at construction (`_discover_sites` +
`_register_hooks`), but those hooks **only record when a capture context is
active** — a `CaptureMode` stack (`TRAINING` or `METRICS`) gates every hook
body. This is deliberate: without it, *any* forward pass through the model
(eval passes, target-network bootstrapping, etc.) would silently pollute the
plasticity statistics. This is also exactly why a monkeypatch/wrapper
approach doesn't work for rl_games — the wrap has to be scoped to precisely
the spans that should count.

- `capture_training()` — used around the actual gradient-computing
  forward+backward pass. Updates both activation metrics (via a forward hook
  on the activation module) and gradient metrics (via a backward hook on the
  producer layer's weight).
- `capture_metrics()` — used around non-training forward passes (e.g.
  rollout) where you still want rank/activity diagnostics but not
  gradient-side (KNIFE) stats — the weight-grad hook is a no-op outside
  `TRAINING` mode.

### Site discovery (`_discover_sites`)

Walks every `nn.Sequential` in the model looking for `Linear -> Activation`
pairs, then looks ahead for the next `Linear` (the "consumer" — needed for
CBP-style utility scoring and for zeroing outgoing weights on reset). A
"feature site" = one activation's output, tied to its producing `Linear` and
consuming `Linear`. Supported activations: ReLU, LeakyReLU, ELU, GELU, SiLU,
Tanh, Sigmoid, `caf.GoLU`.

### `summary()` — diagnostics, called periodically

Returns a flat `dict[str, float]` of metrics at three configurable cadences
(`log_interval`, `rank_interval`, `knife_interval`). Per site, then
aggregated network-wide (unit-count-weighted mean + p10 tails):

- **`dead_units_frac`** / `active_fraction_*` — units gone permanently
  silent ([Dohare2024])
- **`stable_rank` / `effective_rank` / `stable_rank_pct`** — representational
  collapse, an earlier signal than dead units ([Dohare2024])
- **`redo_dormant_units_frac`** — ReDo dormancy score, activation magnitude
  vs. layer mean ([Sokar2023])
- **`contribution_utility_*`** — `|output weight| * |mean feature
  activation|`, bias-corrected EMA; this is also the mechanism used to *rank*
  units for replacement, not just a metric ([Dohare2024] CBP/GnT)
- **`knife_rua_*` / `knife_stagnant_units_frac` / `knife_volatile_units_frac`**
  — gradient-side: is a unit's weight still meaningfully updating relative to
  its own magnitude ([Liu2026])

### `step_replacement()` — CBP/GnT unit replacement, called periodically

Only does anything if `replacement_enabled=True` and `replacement_strategy ==
"cbp"` (the only implemented strategy). Per site with a consumer:

1. **Select units** (`_select_units_by_cbp_utility`) — among units past
   `maturity_threshold` (age), pick the lowest-`contribution_utility` ones.
   Number replaced per step ≈ `replacement_rate * eligible_units`, either
   accumulated deterministically (`replacement_accumulate=True`) or via one
   Bernoulli draw on the fractional part (default).
2. **Reset each unit** (`_reset_unit`):
   - Fold the unit's expected contribution into the consumer layer's bias
     (so removing it doesn't cause a discontinuous jump in output), per CBP
   - Zero the consumer's incoming weight column for that unit
   - Reinitialize the producer's outgoing row via `_initialization_bound`
     (respects `config.init` — `default`/`xavier`/`lecun`/else — and
     `config.activation_name`, which **must match the model's real
     activation** or the reinit gain is wrong)
   - Zero all tracked metric state (utility, activity, redo, knife) for that
     unit index
   - **Zero the optimizer's per-unit state** (`_reset_optimizer_state_for_unit`
     → `_zero_optimizer_state_slice`) — this is the part that requires
     `PlasticityAdam`: it reaches into `optimizer.state[parameter]` and zeros
     the `step`/`exp_avg`/`exp_avg_sq` slices for just that unit's weight
     rows/columns, so a reset unit doesn't inherit stale Adam momentum.

## `PlasticityAdam` — why replacement needs it specifically

`plasticity_adam.py` is a manual reimplementation of Adam (not
`torch.optim.Adam`) with one structural difference: **`state['step']` is a
per-parameter tensor (`torch.zeros_like(p.data)`)**, not a scalar counter
like stock Adam. That's what makes `_zero_optimizer_state_slice` possible —
you can index into `state['step'][unit_idx, :] = 0` the same way as
`exp_avg`/`exp_avg_sq`, so a reset unit's bias-correction restarts from step
0 independently of the rest of the layer. Stock `torch.optim.Adam` uses a
single scalar step count shared by the whole parameter tensor, so per-unit
resets aren't expressible there. Otherwise it's standard Adam math
(bias-corrected first/second moments, optional AMSGrad).

## Key takeaway for the rl_games port

The pattern to replicate for `PlasticityA2CAgent` is exactly this PPO.py
shape, mapped onto rl_games' shared/separate actor-critic trunk instead of
two fully separate networks:
- Build `PlasticityAdam` in place of `Adam` when `replacement_enabled`
- Build one `NetworkPlasticityManager` per trainable sub-network (one if
  `separate: False`/shared trunk, two if `separate: True`)
- Wrap forward+backward in `capture_training()`
- Call `step_replacement()` right after the optimizer step, never before
- Pull `summary()` into whatever logging hook is available (`write_stats`)
