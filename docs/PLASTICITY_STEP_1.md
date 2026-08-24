# Plasticity — Step 1: Add `PlasticityAdam` and `NetworkPlasticityManager`

Part of the larger plasticity integration plan tracked in `ard-isaaclab-tasks`
(`docs/PLASTICITY_INTEGRATION_STEPS.md`). This step lives entirely in this
repo (the `isaac-sim/rl_games` fork, pinned at commit
`6b3534f29568158e9e29ec8bf83cc88fce5f0cae`, branch `plasticity-base`) and
does **not** touch the training loop (`a2c_continuous.py`/`a2c_common.py` —
that's Step 2) or the MLP's own architecture (`network_builder.py` is not
edited). This step only adds two new files:

```
rl_games/algos_torch/plasticity_adam.py
rl_games/algos_torch/plasticity.py
```

Think of it as building an external inspector that gets pointed at an
already-built network, rather than changing the network itself. Setup for
this inspector happens once, at agent construction (`_discover_sites()`
inside `NetworkPlasticityManager.__init__`); everything else it does happens
continuously throughout training, reading/writing into what setup already
found.

Same underlying mechanisms as the reference implementation in
`cares_reinforcement_learning` (`networks/plasticity.py`,
`algorithm/plasticity_adam.py` — see `PLASTICITY_IN_CARES_RL.md` in
`ard-isaaclab-tasks`), but rewritten fresh, not copied — the reference's
code assumes a network shape that doesn't match rl_games' actual structure.

## 1a. Multiple external consumers per site (`output_consumers`)

**The problem**: the reference's site discovery walks a `Linear →
Activation → Linear` pattern by searching *inside* the same `nn.Sequential`
container for what comes after an activation. `nn.Sequential` is a fixed,
ordered list of layers — e.g. `actor_mlp = Sequential(Linear, ELU, Linear,
ELU)`, 4 items. The search loop indexes through that list looking for the
next `Linear` after each activation.

For an interior layer (index 0: `Linear`), the search works — index 1 is
`ELU`, and scanning `children[2:]` finds another `Linear` at index 2.
Consumer found, site built normally.

For the *last* hidden layer (index 2: `Linear`), index 3 is `ELU`, but
scanning `children[4:]` finds **nothing** — the list has ended. Not because
the search is broken, but because there's genuinely nothing left in that
list to find.

rl_games' actual output heads — `mu`, `sigma`, `value` — are **not** inside
`actor_mlp`'s `Sequential` at all. They're separate attributes on the outer
network class (`self.mu = nn.Linear(...)`, sitting beside `self.actor_mlp`,
not inside it):
```python
class A2CBuilder.Network:
    self.actor_mlp = nn.Sequential(...)      # network_builder.py:214-215,276
    self.mu = torch.nn.Linear(out_size, actions_num)   # :291 — separate attribute
    self.value = self._build_value_layer(...)          # :280 — separate attribute
    ...
    a_out = self.actor_mlp(a_out)             # :399
    mu = self.mu_act(self.mu(a_out))          # :413 — reads a_out directly, outside the Sequential
```
The search loop only ever iterates `actor_mlp.named_children()` — it has no
code path that looks at `self.mu`/`self.value` at all, regardless of how
thoroughly it searches, because they were never part of the list being
searched.

**The fix**: let `FeatureSite` take a *tuple* of external consumer modules,
supplied explicitly at construction time (when the caller already has a
reference to the real network object) instead of relying purely on search:
```python
@dataclass
class FeatureSite:
    name: str
    producer_module: nn.Linear
    hook_module: nn.Module
    consumer_modules: tuple[nn.Linear, ...]
```
```python
NetworkPlasticityManager(
    model=network.actor_mlp,
    optimizer=optimizer,
    output_consumers=[network.mu, network.sigma, network.value],  # shared trunk (separate: False)
)
```
Only the site where in-`Sequential` search finds nothing falls back to using
`output_consumers`; every interior site is unaffected and keeps using the
normal search. Reset logic (`_reset_unit`) loops over every consumer in
`consumer_modules`, folding bias and zeroing the relevant weight
column/row in **each** — since all of them genuinely depend on that same
unit's output, not just one.

**`separate: true` does not avoid this problem** — considered and rejected.
Two independent reasons:
1. It doesn't fix anything: even with two separate networks, `actor_mlp`
   still branches into `mu` **and** `sigma` (two heads, same "outside the
   box" problem, just smaller); `critic_mlp`'s `value` head is still built
   outside its `Sequential` too, regardless of the flag.
2. It changes the actual architecture being trained — `separate: true`
   duplicates the entire trunk (two independent `[hidden]`-sized MLPs
   instead of one shared one), roughly doubling hidden-layer parameter
   count, and would make new runs architecturally incomparable to existing
   `cartpole`/`shadow_hand` baselines trained with `separate: false`.

**Rank metrics use the rollout activation window specifically** (stable
rank / effective rank / stable-rank percentage), not the training-capture
window — rollout and training activations have different statistics
(different batch composition, `eval()` vs. training-mode forward), so keep
this source distinction explicit in the manager rather than mixing windows.

**Caveat, resolved**: `self.sigma` is not always a `Linear`. With
`fixed_sigma` config, it's a raw `nn.Parameter`
(`network_builder.py:298`) that doesn't consume `a_out` at all —
`sigma = mu * 0.0 + self.sigma_act(self.sigma)` (line 415), a free-standing
vector, not a weight matrix reading the trunk's output. Checked both tasks'
YAML configs in `ard-isaaclab-tasks` directly
(`agents/rl_games_ppo_cfg.yaml`, `model.network.space.continuous`): **both
`cartpole` and `shadow_hand` set `fixed_sigma: True`**. So for both tasks,
`output_consumers = [network.mu, network.value]` — `sigma` excluded for
both, not a per-task check.

## 1b. Logging cadence — call `summary()` once per epoch, same as the reference

**Revised** (superseded the original frame-boundary-gating design below, kept
struck through for history). Traced how the reference actually calls
`summary()`: `training_runner.py::run_training()` only invokes
`self.agent.train()` once per `number_steps_per_train_policy` env steps
(default 10000, gated at the top of the loop) — i.e. once per "epoch" in the
reference's own terms, flushing the memory buffer each time. Inside
`PPO.py::update_from_batch`, the `summary()` calls (lines 649-650) sit at the
end of that same once-per-epoch call, after the mini-epoch/minibatch loop —
not once per minibatch, not once per env step.

rl_games' `write_stats()` is also only called once per epoch (`train()`'s
main loop), never once per minibatch/optimizer step. So the two codebases'
call frequency for `summary()` already matches directly — there's no
mismatch to correct with an extra gating layer. `curr_frames` (the number of
env frames one epoch produces) is fixed for the whole run
(`horizon_length × num_actors × num_agents`), so "once per K epochs" and
"once per K×curr_frames frames" are the same schedule, not just similar.

Fix: don't add a separate frame-boundary-crossing counter inside the
manager. Call `manager.summary(...)` once per epoch from `write_stats`
(matching the reference's call site 1:1) and log whatever dict it returns
straight through `self.writer.add_scalar(...)`, keyed by `frame` (the value
`write_stats` already has on hand) purely as the **x-axis tag** — not as a
gating condition. `log_interval`/`rank_interval`/`knife_interval` stay as
the reference implements them: an internal counter incremented once per
`summary()` call (i.e. once per epoch, same cadence either codebase), same
as `cares_reinforcement_learning`'s `step_count`.

<details>
<summary>Superseded: original frame-boundary-gating design</summary>

Lives in the run-time (continuous) part of the manager, inside `summary()`.
Recomputing every diagnostic metric — especially rank, the expensive one —
on every single call would be wasteful. Two independent counters gate this:
- `optimizer_steps` — incremented inside `on_optimizer_step()`, so once per
  minibatch (the training phase of the loop).
- `environment_frames` — passed into `summary(frame=frame)` from
  `write_stats`, so once per epoch.

`summary()` checks `environment_frames % log_interval_frames` and
`% rank_interval_frames`; if neither threshold is crossed it returns `{}`
that call — most calls are cheap no-ops, only every Nth actually recomputes
and logs. Frame count is used (not a raw internal step counter, as the
reference does) because it's Isaac Lab's natural progress unit, and one
optimizer step corresponds to a variable, config-dependent number of
environment frames.

This assumed `summary()` might be called more often than once per epoch
(e.g. per minibatch) — checked against the actual call sites in both
codebases and that assumption doesn't hold; superseded above.

</details>

## 1c. Rollout activation subsampling (`rollout_samples_per_forward`)

Lives even deeper in the run-time path than 1b — inside the **forward
hook** itself, fired during every `capture_rollout()` span (i.e. every
`get_action_values()` call, every rollout step).

Isaac Lab runs thousands of parallel environments — a rollout forward pass
processes all of them at once, so the activation the hook sees is
`[num_envs, hidden]` (e.g. `[4096, hidden]`), not `[1, hidden]`. Recording
every row on every rollout step would fill the behavioural window almost
instantly (`4096 envs × 32 rollout steps ≈ 131k rows` from a single
rollout) and cost real GPU→CPU transfer overhead on every forward pass.

Fix: inside the hook, before appending to the window, randomly subsample a
fixed number of environments:
```yaml
rollout_samples_per_forward: 256
```
Lower transfer cost, and the window ends up spanning more wall-clock time
(sampled across many rollouts) rather than being dominated by one instant.
Treat this as an experimental parameter, not a fixed default.

## 1d. Scope for the first implementation

Setup-time, same `_discover_sites()` call as 1a — a guard on what counts as
a valid site, not an addition to what's found. Deliberately target only:
continuous PPO, standard MLP (`nn.Sequential` of `Linear` + activation),
non-RNN, non-D2RL. Reject or skip: RNN, D2RL, CNN, trainable normalization
(`LayerNorm` etc.) inside the managed MLP — if one of these sits between an
activation and the next `Linear`, the search must recognise it can't handle
that shape and exclude the site, rather than silently misinterpreting it as
`Linear → Linear`.

`cartpole` and `shadow_hand` are both plain MLPs (no RNN/CNN/LayerNorm), so
this guard doesn't exclude anything for either task currently — it's a
safety net for network shapes this code isn't designed for yet, not
something either task will actually hit.

## How 1a-1d relate

| Sub-task | When it runs | What it governs |
|---|---|---|
| 1a | Once, at construction (`_discover_sites`) | What counts as a site, and what its consumer(s) are |
| 1d | Once, at construction (`_discover_sites`) | What gets excluded from being a site at all |
| 1b | Every `summary()` call (once per epoch, same cadence as the reference's `step_count`) | How often expensive diagnostics (rank, KNIFE) actually get computed |
| 1c | Continuously, every rollout step (forward hook) | How much rollout data gets recorded per step |

1a and 1d together fully determine `self.sites` — the fixed list every other
piece of plasticity code (hooks, `step_replacement`, `summary`) reads from
for the rest of the run. 1b and 1c don't change *what* is being watched,
only *how much*/*how often* data gets recorded and reported.

## Recommended order within Step 1

1. `PlasticityAdam` first (`plasticity_adam.py`) — no dependency on the
   manager, simplest piece, needed by the manager's replacement path later.
2. `NetworkPlasticityManager` (`plasticity.py`) with `output_consumers`
   support from the start (1a) — don't bolt it on after the fact.
3. Add 1d's scope guards to the same discovery pass.
4. Add 1b's `step_count`-based interval gating to `summary()` (mirrors the
   reference directly — call site frequency already matches, see 1b above).
5. Add 1c's subsampling to the forward hook.
6. **Before moving to Step 2**: run a standalone site-discovery test against
   the real `cartpole`/`shadow_hand` network shapes — print discovered
   sites, confirm every intended hidden layer (including the last one, via
   `output_consumers=[mu, value]`) is found.
