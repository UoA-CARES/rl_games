# Plasticity Integration Steps

Scope: **plasticity becomes a native `rl_games` feature.** `a2c_continuous.py`
and `a2c_common.py` are edited directly (in place), instead of being
subclassed from a new file in this repo. `PlasticityAdam` and
`NetworkPlasticityManager` are added as new files inside `rl_games/algos_torch/`.

- `cares_reinforcement_learning` — reference/conceptual only. Nothing gets
  imported from it and nothing in it is edited. See `PLASTICITY_IN_CARES_RL.md`
  for how that reference implementation works — the mechanisms (capture
  modes, CBP replacement, per-unit Adam state) are the same; the concrete
  code is a fresh implementation adapted to `rl_games`' actual network shape
  (see Step 1).
- `rl_games` (installed package) — **edited directly** (decision superseded
  the earlier subclass plan; see "Why direct edit, not a subclass" below).
  **Resolved**: the installed `rl_games==1.6.1` is **not** upstream
  `Denys88/rl_games` — its `direct_url.json` shows it was installed from
  **`https://github.com/isaac-sim/rl_games.git`** (NVIDIA's Isaac Sim fork)
  at commit `6b3534f29568158e9e29ec8bf83cc88fce5f0cae`. The public
  `Denys88/rl_games` `v1.6.1` tag is a **different codebase** — diffing it
  against the installed copy showed real discrepancies (missing
  `restore_central_value_function`, missing the aux-loss branching logic
  cited in Step 2c). Cloned the correct fork to
  `~/Documents/Github/rl_games`, checked out that exact commit on branch
  `plasticity-base`, and verified byte-identical against the installed copy
  for `a2c_continuous.py`, `a2c_common.py`, `network_builder.py`, and
  `models.py`. Installed editable into `env_isaaclab`
  (`pip install -e ~/Documents/Github/rl_games --no-deps`) — `rl_games.__file__`
  now resolves to the local clone, confirmed via `direct_url.json`
  (`"editable": true`). All future edits happen in that clone; commit there
  as normal git history.
  **Note**: this fork has no `rl_games.__version__` attribute at all
  (`AttributeError` on access) — the version-coupling guard planned for Step
  2f needs to pin against the git commit hash instead, not a version string.
- `ard-isaaclab-tasks` — the `--plasticity` flag (Step 3) and per-task YAML
  config (Step 4) still live here, same as before.

Covers `cartpole` and `shadow_hand` only (the two tasks already used for
testing). Verified against the actually-installed `rl_games==1.6.1`
(`/home/harsh/env_isaaclab/lib/python3.11/site-packages/rl_games`).

## Why direct edit, not a subclass

The original plan (superseded) subclassed `A2CAgent` from a new file in this
repo, overriding 5 methods. Four of those were genuine extensions
(`__init__`, `get_action_values`, `trancate_gradients_and_step`,
`write_stats` — each called `super()` and added a bit on top, so they'd
automatically track upstream rl_games behaviour). One, `calc_gradients`,
could not be — rl_games' real implementation does the entire forward pass +
loss + `.backward()` inline with no seam to hook into, so wrapping the
forward/backward span in `capture_training()` required copying the *entire*
method into the subclass and editing the copy.

That copy is a permanently frozen duplicate with no tooling to check it
against future rl_games versions — you'd have to manually re-read the new
source and eyeball-diff it against the copy. This already caused a real bug
during planning: the aux-loss accumulation logic (see Step 2b below) was
initially copied in simplified form and only caught by re-checking against
the installed source.

Editing `a2c_continuous.py`/`a2c_common.py` directly turns that one
unavoidable fork into a small, in-place, git-diffable change — future
rl_games upgrades can be checked with normal diff/rebase tooling instead of
manual comparison. It also removes the asymmetry where 4/5 changes were
"clean" and 1/5 was a landmine.

## Step 1 — Add `PlasticityAdam` and `NetworkPlasticityManager` to rl_games

New files:
```
rl_games/algos_torch/plasticity_adam.py
rl_games/algos_torch/plasticity.py
```

Same mechanisms as the `cares_reinforcement_learning` reference — see
`PLASTICITY_IN_CARES_RL.md` — but the site-discovery logic needs a real
structural change for rl_games' actual network shape, found by reading
`rl_games/algos_torch/network_builder.py` directly:

### 1a. Multiple external consumers per site (`output_consumers`)

The reference's site discovery only recognises a site if the *next module in
the same `nn.Sequential`* is another `Linear` (the "consumer," used to fold
bias and zero weight columns on reset). rl_games' actual `A2CBuilder`
(`network_builder.py`) doesn't end its actor/critic trunks that way:

```python
self.actor_mlp = nn.Sequential(...)      # network_builder.py:214-215,276
...
self.mu = torch.nn.Linear(out_size, actions_num)   # :291 — separate module
self.value = self._build_value_layer(...)          # :280 — separate module
...
a_out = self.actor_mlp(a_out)             # :399
mu = self.mu_act(self.mu(a_out))          # :413 — reads a_out directly
```

`mu`, `sigma`, and `value` sit **outside** `actor_mlp`'s `Sequential`, each
reading the trunk's final hidden layer directly. Under the reference's
"next thing in the Sequential" logic, the trunk's *last* hidden layer would
never find an in-sequence consumer and would silently be skipped —
permanently excluding its units from both diagnostics and replacement.

Fix: let `FeatureSite` take a **tuple** of external consumer modules instead
of a single one:
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
    output_consumers=[network.mu, network.sigma, network.value],  # shared trunk
)
```
Reset logic (`_reset_unit`) loops over every consumer in `consumer_modules`,
folding bias and zeroing the relevant weight column/row in **each** — since
all of them genuinely depend on that same unit's output, not just one.

**Rank is computed from the rollout activation window specifically** (stable
rank / effective rank / stable-rank percentage), not the training-capture
window — keep this source distinction explicit in the manager, since rollout
and training activations can have different statistics (different batch
composition, `eval()` vs. training-mode forward).

**Caveat, resolved**: `self.sigma` is not always a `Linear`. With
`fixed_sigma` config, it's a raw `nn.Parameter` (`network_builder.py:298`)
that doesn't consume `a_out` at all — `sigma = mu * 0.0 +
self.sigma_act(self.sigma)` (line 415), a free-standing vector, not a
weight matrix reading the trunk's output. Checked both tasks' YAML configs
directly: **both `cartpole` and `shadow_hand` set `fixed_sigma: True`**
(`agents/rl_games_ppo_cfg.yaml`, `model.network.space.continuous`). So for
both tasks, `sigma` must **not** be included in `output_consumers` — there's
no weight column tied to trunk units to reset for it.
`output_consumers = [network.mu, network.value]` for both.

### 1b. Frame-based logging cadence, separate from optimizer-step cadence

The reference's `PlasticityConfig` gates diagnostics off an internal
`step_count` incremented once per `summary()` call. For Isaac Lab, track two
independent counters instead:
- `optimizer_steps` — drives `step_replacement()`'s cadence (still once per
  minibatch/optimizer update, same as the reference).
- `environment_frames` — drives `summary()`'s logging cadence
  (`log_interval_frames`, `rank_interval_frames`), since frame count is
  Isaac Lab's natural progress unit and one optimizer step corresponds to a
  variable, config-dependent number of environment frames.

### 1c. Rollout activation subsampling (`rollout_samples_per_forward`)

Isaac Lab runs thousands of parallel environments — a single rollout
activation tensor is `[num_envs, hidden]` (e.g. `[4096, hidden]`), not
`[1, hidden]`. Feeding every environment's activation into the behavioural
window fills it almost instantly (4096 envs × 32 rollout steps ≈ 131k rows
from one rollout) and adds GPU→CPU transfer overhead on every forward pass.

Fix: randomly subsample a fixed number of environments per rollout forward
before adding to the window:
```yaml
rollout_samples_per_forward: 256
```
Lower transfer cost, and the window ends up spanning more wall-clock time
(sampling across many rollouts) rather than being dominated by one instant.
Treat this as an experimental parameter, not a fixed default.

### 1d. Scope for the first implementation

Deliberately target only: continuous PPO, standard MLP (`nn.Sequential` of
`Linear`+activation), non-RNN, non-D2RL. Reject or defer: RNN, D2RL, CNN,
trainable normalization (LayerNorm etc.) inside the managed MLP — site
discovery should skip past those rather than silently misinterpreting them
as `Linear → Linear`. `cartpole` and `shadow_hand` are both standard MLPs,
so this isn't a blocker for either task.

## Assessed difficulty

| Scope | Difficulty |
|---|---:|
| Diagnostics only | ~2/10 |
| Diagnostics matching the existing plasticity semantics | ~4/10 |
| Full CBP/GnT-style replacement | ~5/10 |
| Clean long-term rl_games integration | ~6/10 |

Not PPO or Isaac Lab itself — the difficulty is concentrated in: rl_games'
network/head structure (Step 1a), capturing rollout activations without
overwhelming CPU memory (Step 1c), correctly resetting Adam state on
replacement, maintaining throughput when swapping fused Adam for
`PlasticityAdam`, eventually synchronizing replacement decisions for
multi-GPU, and persisting plasticity state in checkpoints.

## Step 2 — Wire plasticity into the real rl_games classes

### 2a. `a2c_common.py` — one generic hook, kept plasticity-agnostic

The manager exposes its capture spans as `manager.capture_rollout()` /
`manager.capture_training()` (renamed from the earlier `capture_metrics()`
to match the latest naming), plus `manager.on_optimizer_step()` for
replacement. Add a single no-op method to the shared base class so *other*
rl_games algorithms (SAC, DQN, etc.) aren't coupled to plasticity just
because this file changed:
```python
def plasticity_rollout_context(self):
    return nullcontext()
```
`A2CAgent` (below) overrides it to enter every attached manager's
`capture_rollout()`; everything else in `a2c_common.py` calls this generic
hook without knowing what plasticity is.

### 2b. `get_action_values` (`a2c_common.py:410-430`) — rollout capture

Wrap the existing forward pass in the (overridden) rollout context. The
method already runs in `eval()` mode (`self.model.eval()`, line 412), so
this reflects true rollout behaviour — matches the reference's `act()` wrap:
```python
def get_action_values(self, obs):
    ...
    with self.plasticity_rollout_context(), torch.no_grad():
        res_dict = self.model(input_dict)
    ...
```
(`plasticity_rollout_context()`, on `A2CAgent`, enters `mgr.capture_rollout()`
for each attached manager.)
Called from `play_steps()`/`play_steps_rnn()` (`a2c_common.py:761,833`).

**Do not** wrap `get_values` (`a2c_common.py:432-455`) — used for GAE
bootstrapping (`last_values = self.get_values(self.obs)`, called at
`a2c_common.py:800,881`). Auxiliary value-bootstrap computation, not real
policy behaviour — excluding it from capture matches the reference
implementation's own reasoning about not polluting behavioural stats with
non-representative forward passes.

### 2c. `calc_gradients` (`a2c_continuous.py:93-190`) — training capture

Wrap the existing forward pass + `.backward()` in the training capture
context (`plasticity_training_context()` enters `mgr.capture_training()` for
each attached manager) — same lines, same logic, just indented one level
inside a `with`:
```python
with self.plasticity_training_context():
    res_dict = self.model(batch_dict)
    ...
    loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
    ...
    self.scaler.scale(loss).backward()
self.trancate_gradients_and_step()
```

**Known discrepancy to preserve, not simplify**: the installed 1.6.1
aux-loss accumulation is genuinely branching —
```python
if k in self.aux_loss_dict:
    self.aux_loss_dict[k] = v.detach()
else:
    self.aux_loss_dict[k] = [v.detach()]
```
not a flat `self.aux_loss_dict[k] = v.detach()`. Since this is now an
in-place edit rather than a copied fork, this logic is naturally preserved
verbatim (it's the surrounding code, untouched) — this note exists mainly
as a reminder of why the file must be edited, not rewritten.

### 2d. `trancate_gradients_and_step` (`a2c_common.py:324-347`) — replacement timing

Add the replacement call after the real optimizer step:
```python
def trancate_gradients_and_step(self):
    ...
    self.scaler.step(self.optimizer)
    self.scaler.update()
    if self.plasticity_enabled and self.plasticity_replacement_enabled:
        for mgr in self.plasticity_managers:
            mgr.on_optimizer_step()
```
(`on_optimizer_step()` — renamed from `step_replacement()` to match the
latest manager API — increments the optimizer-step counter and runs CBP
replacement if due.)
`self.scaler.step(self.optimizer)` is the single choke point where the
optimizer actually updates weights — running replacement immediately after
ensures a reset unit's fresh weights are never touched by a stale
pre-reset gradient still in flight.

**Note on cadence**: unlike the reference implementation (separate actor and
critic optimizers, replacement called once per network), rl_games normally
runs actor and critic through one **combined** PPO loss and one shared
optimizer step (see `calc_gradients`, Step 2c). The native integration
should retain that — replacement fires once per combined optimizer update,
not once per network, even when `output_consumers` differ between the actor
and critic heads.

### 2e. `write_stats` (`a2c_common.py:360-380`) — logging

Append plasticity metrics to the existing scalar-writing loop, keyed by
frame (not the internal step counter — see Step 1b):
```python
def write_stats(self, total_time, epoch_num, step_time, play_time, update_time,
                 a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                 scaled_time, scaled_play_time, curr_frames):
    ...  # existing scalars, unchanged
    if self.plasticity_enabled:
        for mgr in self.plasticity_managers:
            for key, value in mgr.summary(frame=frame).items():
                self.writer.add_scalar(f"plasticity/{mgr.name}/{key}", value, frame)
```

### 2f. `A2CAgent.__init__` (`a2c_continuous.py`) — construction

- Read the `plasticity:` block from config, build a config object if
  `enabled`.
- Build the model/optimizer as normal first.
- If `replacement_enabled`, swap the optimizer construction for
  `PlasticityAdam(self.model.parameters(), lr=..., eps=1e-8, weight_decay=...)`.
  Otherwise keep rl_games' normal (fused) `Adam` — diagnostics-only mode
  should not pay the custom-optimizer cost (see Step 5, Stage 1 vs. 2).
- Attach manager(s) per Step 4c below (shared vs. separate critic).
- Guard: reject/raise on `multi_gpu=True` when `replacement_enabled` —
  independent ranks could pick different units to replace, desyncing the
  model even with gradients synchronized.
- Guard: this fork has no `__version__` attribute to assert against (see
  the resolved scope note at the top). Instead, pin/document the expected
  git commit (`6b3534f29568158e9e29ec8bf83cc88fce5f0cae` at time of writing)
  in the clone itself (e.g. a tag or a checked-in `PLASTICITY_BASE_COMMIT`
  note), so a future rebase/upgrade of the fork is a deliberate, visible
  step rather than a silent structural drift.

## Step 3 — Wire the `--plasticity` flag into `scripts/train.py`

Unchanged from the original plan — still lives entirely in this repo, still
opt-in, still resolves to plain rl_games behaviour when the flag is absent
(the patched `A2CAgent` should early-return to identical behaviour when
`plasticity.enabled` is false, so the flag genuinely gates all of it, not
just registration):

```python
parser.add_argument("--plasticity", action="store_true", default=False,
                     help="Enable plasticity monitoring/unit-replacement.")
```
```python
if args_cli.plasticity:
    agent_cfg["params"]["config"].setdefault("plasticity", {})["enabled"] = True
```
No factory/registry swap needed anymore — since `A2CAgent` itself now
understands plasticity, there's no separate `PlasticityA2CAgent` class to
register under a new algo name. The flag just forces `plasticity.enabled`
on regardless of what the YAML says.

## Step 4 — Add a `plasticity:` config block per task

Per-task block under `params.config`, now with the frame-based and
subsampling fields from Step 1:

```yaml
plasticity:
  enabled: true
  replacement_enabled: false
  replacement_rate: 1.0e-5
  maturity_threshold: 1000
  replacement_accumulate: false
  utility_decay: 0.99
  activity_threshold: 1.0e-5
  dormant_threshold: 0.1
  stagnant_threshold: 0.25
  volatile_threshold: 4.0
  rua_eps: 1.0e-8
  activation_window_size: 10000
  rollout_samples_per_forward: 256
  compute_rank: true
  log_interval_frames: 100000
  rank_interval_frames: 1000000
  knife_interval_summaries: 1
  init: kaiming
  activation_name: elu   # both tasks use ELU — config default is relu, must override
```

Both tasks use `separate: False` (shared trunk) → one manager, name
`"shared"`, `output_consumers=[network.mu, network.value]` — **not**
`sigma`, since both tasks set `fixed_sigma: True` (resolved in Step 1a).
Tune
`replacement_rate`/`maturity_threshold`/interval values per task the same
way the original plan did — `cartpole`'s short run (150 epochs) needs a
much more aggressive schedule than `shadow_hand` (5000 epochs) or
replacement will never trigger.

## Step 5 — Run the staged comparison

Same underlying goal as before, restructured into the more explicit staged
experiment progression:

**Stage/Experiment A — baseline.** Stock PPO, no plasticity code active at
all (`enabled: false` or flag absent).

**Stage/Experiment B — diagnostics only.** `enabled: true,
replacement_enabled: false` — stock fused Adam still used (cheap). Confirms
(1) the forward/backward and rollout wraps are correct — fitness curve
should be visually indistinguishable from A; (2) whether plasticity loss is
actually observable in these two tasks at all, via `dead_units_frac`,
`stable_rank_pct`, `knife_stagnant_units_frac`.

**Stage/Experiment C — full CBP.** `replacement_enabled: true` — switches to
`PlasticityAdam`. Confirms `plasticity/<name>/units_replaced_total` rises
above 0 (retune `maturity_threshold`/`replacement_rate` if it stays 0 on
cartpole's short run), and whether replacement changes task fitness
vs. B.

**Benchmark** (run alongside/after C): compare training FPS, wall-clock
training time, learning curves, final task performance, stable rank,
effective rank, dead/dormant units, utility distribution, and replacement
rate between A/B/C — this is what actually answers whether plasticity is
worth its throughput cost, not just whether it runs.

**Experiment D — later ablations** (not required for the initial pass):
ReDo-only, KNIFE-only, or combinations, once A/B/C are validated.

```bash
./isaaclab.sh -p scripts/train.py --task <task> --seed 42 --num_envs <N> --max_iterations <M>
./isaaclab.sh -p scripts/train.py --task <task> --seed 42 --num_envs <N> --max_iterations <M> --plasticity
```

## Recommended implementation order

1. `PlasticityAdam`
2. `NetworkPlasticityManager` (with `output_consumers` support from the start)
3. Attach manager(s) to actor/critic MLP
4. **Run a standalone site-discovery test** — print discovered sites,
   confirm every intended hidden layer (including the last one, via
   `output_consumers`) is found, before wiring anything else up
5. Rollout capture (`get_action_values`)
6. Training capture (`calc_gradients`)
7. TensorBoard summaries (`write_stats`)
8. Run Experiment B (diagnostics-only), verify metrics look sane
9. Enable `PlasticityAdam`
10. Enable CBP replacement, run Experiment C
11. Benchmark throughput cost of `PlasticityAdam` vs. fused Adam
12. Add plasticity manager state to checkpointing (see risks below)
13. Multi-GPU replacement synchronization — later work, not required now

## Known risks / open verification items

- **Deployment/reproducibility of the patched rl_games** — resolved. Fork
  cloned to `~/Documents/Github/rl_games` (from `isaac-sim/rl_games`, not
  upstream `Denys88/rl_games` — they diverge), pinned to commit
  `6b3534f29568158e9e29ec8bf83cc88fce5f0cae`, installed editable into
  `env_isaaclab`. Anyone else running this repo needs the same clone +
  editable install, not a plain `pip install rl-games`.
- **No `rl_games.__version__` on this fork** — the version-coupling guard in
  Step 2f pins against the git commit instead (see Step 2f).
- ~~`fixed_sigma` vs. `output_consumers`~~ — resolved: both `cartpole` and
  `shadow_hand` set `fixed_sigma: True`, so `sigma` is excluded from
  `output_consumers` for both (see Step 1a).
- **Version-coupled edits** — the `get_action_values`/`calc_gradients`
  patches are tied to the actually-installed `1.6.1` structure; the
  `__version__` assertion (Step 2f) makes a future rl_games bump fail
  loudly rather than silently diverge.
- **Multi-GPU isn't handled** — guarded against when `replacement_enabled`.
- **`PlasticityAdam` fully replaces fused Adam when replacement is on** —
  smoke-test one short `replacement_enabled: true` run to confirm
  `lr_schedule: adaptive` (reads/writes `optimizer.param_groups[...]['lr']`)
  still works against it, and benchmark the throughput cost (Step 11).
- **Numerical validation of `PlasticityAdam`** — compare against stock
  `torch.optim.Adam` on a small network before trusting its update math.
- **Mixed precision** — rl_games uses `torch.cuda.amp.autocast`; hook
  behaviour (forward/backward hooks recording under autocast) needs testing
  under the exact mixed-precision config used, not assumed to work.
- **Checkpointing gap** — rl_games' checkpointing saves model + optimizer
  but not plasticity manager state. Full list of state that would need to be
  included: `optimizer_steps`, neuron ages, utility EMA, bias-correction
  state, ReDo EMA state, KNIFE/lifetime state, replacement accumulator,
  replacement counters, reporting state. Resumed runs currently lose all of
  this silently. Needs addressing if either task's runs are ever resumed
  from checkpoint.
- **Config duplication** — no shared base YAML between tasks, so any future
  tuning of shared plasticity defaults must be kept in sync by hand.
