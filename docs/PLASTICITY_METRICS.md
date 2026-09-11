# Plasticity Metrics Reference

`rl_games/algos_torch/plasticity.py` implements `NetworkPlasticityManager`,
which attaches to the actor/critic/shared trunk of a PPO network and tracks
neuron-level "plasticity loss" diagnostics — dead units, feature-rank
collapse, dormant neurons, and gradient-side stagnation. Its `summary()`
method is the *only* source of every metric under the `plasticity/` tag
prefix in TensorBoard: `rl_games/common/a2c_common.py` calls
`mgr.summary()` for each active manager and writes every returned key/value
pair as `self.writer.add_scalar(f'plasticity/{key}', value, frame)`.

This doc explains every metric that can appear under `plasticity/...`. It
does not cover the rest of the training run's TensorBoard tags
(`losses/*`, `rewards/*`, `info/*`, `performance/*`, `diagnostics/*`, etc.)
— those are logged directly inline in the training loop, not through
`summary()`.

## How a network gets tracked

`NetworkPlasticityManager` discovers "feature sites": `Linear -> Activation`
pairs inside an `nn.Sequential` trunk, each tied to the `Linear` that
produces the activation and the `Linear`(s) that consume it downstream
(`_discover_sites`). One manager is built per trainable sub-network
(`init_plasticity()` in `a2c_common.py`):

- `separate: True` network config → two managers, named `"actor"` and
  `"critic"`.
- `separate: False` (shared trunk) → one manager, named `"shared"`.

Every metric is reported twice: once **per site** (per hooked layer) and
once **network-wide** (aggregated across all of that manager's sites).

### Tag naming

| Scope | Tag pattern | Example |
|---|---|---|
| Per-site / per-layer | `plasticity/{network}/{site}/{metric}` | `plasticity/actor/actor_mlp_0/redo_dormancy_score_p10` |
| Network-wide | `plasticity/{network}/{metric}` | `plasticity/actor/redo_dormant_units_frac` |

- `{network}` = `actor`, `critic`, or `shared`.
- `{site}` = the producing layer's dotted module path with `.` replaced by
  `_` (e.g. `actor_mlp.0` → `actor_mlp_0`).
- `{metric}` = one of the metric keys documented below.

### Logging cadence

`summary()` is called once per epoch, but most metrics don't appear on every
call — three independent counters gate expensive diagnostics:

| Config key | Default | Gates |
|---|---|---|
| `log_interval` | 10 | Most metrics (`should_log`) — recent-activity window, ReDo, contribution utility, weight magnitude |
| `rank_interval` | 50 | SVD-based rank metrics (`should_rank`) — `stable_rank`, `effective_rank`, `stable_rank_pct` |
| `knife_interval` | 1 (multiple of `log_interval`) | KNIFE gradient metrics (`should_knife`) |

Other constructor hyperparameters that change *values*, not cadence, and are
referenced throughout this doc:

| Config key | Default | Used by |
|---|---|---|
| `utility_decay` | 0.99 | EMA decay for ReDo activation magnitude, ReDo activity fraction, and contribution utility |
| `activity_threshold` | 1e-5 | Minimum `\|activation\|` counted as "active" for the ReDo activity-fraction EMA |
| `dormant_threshold` | 0.1 | ReDo dormancy-score cutoff for `redo_dormant_units_frac` |
| `stagnant_threshold` | 0.25 | KNIFE RUA cutoff for `knife_stagnant_units_frac` |
| `volatile_threshold` | 4.0 | KNIFE RUA cutoff for `knife_volatile_units_frac` |
| `rua_eps` | 1e-8 | Numerical floor in the KNIFE update-activity denominator |
| `activation_window_size` | 10000 | Max rows kept in the rolling activation buffer used for dead-unit/rank stats |

Two distinct kinds of "recent vs. long-run" show up below and are easy to
conflate:

- **Window metrics** (`dead_units_frac`, `active_fraction_*`, plain
  `knife_*`) are computed from state that is reset every time `should_log`
  (or `should_knife`) fires — they describe behaviour since the *last* log.
- **Lifetime metrics** (`*_lifetime_*`) are cumulative running averages
  since training started (or the last checkpoint restore) and never reset.
- **EMA metrics** (ReDo's `redo_*`, contribution utility) are neither — they
  are continuously decayed (`utility_decay`) since the start of training and
  never reset, so they sit between the two: recency-weighted, but not tied
  to the logging window.

## References

The diagnostics in `plasticity.py` reimplement metrics from three papers:

- **[Dohare2024]** Dohare, S., Hernandez-Garcia, J.F., Lan, Q., Rahman, P.,
  Mahmood, A.R., & Sutton, R.S. (2024). "Loss of Plasticity in Deep
  Continual Learning." *Nature*, 632, 768–774.
  → dead units, stable rank / effective rank (Fig. 2d, 4b), CBP/GnT
  contribution-utility replacement.
- **[Sokar2023]** Sokar, G., Agarwal, R., Castro, P.S., & Evci, U. (2023).
  "The Dormant Neuron Phenomenon in Deep Reinforcement Learning." ICML 2023.
  → ReDo dormancy score and dormant-unit fraction.
- **[Liu2026]** Liu, Z., Gao, Z., Qin, H., Hu, J., Wu, J., Zhu, M., Zhang,
  H., Ma, C., Shen, S., & Wang, C. (2026). "Stagnant Neuron: Towards
  Understanding the Plasticity Loss in Multi-Agent Reinforcement Learning
  Value Factorization Methods." arXiv:2606.25335.
  → KNIFE relative update activity (RUA), stagnant/volatile neuron
  fractions.

---

## Per-Site / Per-Layer Metrics

These are computed once per tracked feature site (hooked layer) and logged
under `plasticity/{network}/{site}/{metric}`.

### Dead / active units — [Dohare2024]

**Key idea:** the most literal symptom of plasticity loss — a unit whose
activation has gone essentially silent contributes nothing to the network
anymore. Computed two ways: over a short rolling window (resets every
`log_interval`) and cumulatively over the network's whole lifetime.

Recent-window metrics (`_summarize_recent_activity`), from a buffer of up to
`activation_window_size` raw activation rows:

```python
active_fraction = (activity_window > 0.0).float().mean(dim=0)  # per unit
```

| Metric | Meaning |
|---|---|
| `dead_units_frac` | Fraction of units in this layer with `active_fraction < 0.01` — i.e. fired on fewer than 1% of the buffered rows. |
| `active_fraction_mean` | Mean, across units, of the fraction of buffered rows on which each unit fired at all. |
| `active_fraction_p10` | 10th percentile of `active_fraction` across units — 90% of units in the layer fire more often than this. |
| `activity_window_size` | Number of rows currently buffered (diagnostic, not a plasticity signal by itself — a small value means the other metrics in this group are noisy). |

Lifetime metrics (`_summarize_lifetime_activity`), from a running average
that accumulates every training batch and never resets:

```python
active_fraction = lifetime.total / lifetime.count.clamp_min(1.0)
```

| Metric | Meaning |
|---|---|
| `dead_units_lifetime_frac` | Fraction of units with lifetime `active_fraction < 0.01` — units dead not just recently, but for (nearly) the whole run. |
| `active_lifetime_fraction_mean` | Mean lifetime active-fraction across units. |

### Feature rank — [Dohare2024]

**Key idea:** dead-unit counting only catches units that have gone
*completely* silent. A layer can lose useful capacity earlier than that, by
having many units collapse into near-duplicates of each other — still
firing, but no longer spanning independent directions. Rank metrics measure
that directly via the SVD of the buffered activation window
(`_rank_metrics`, gated by `rank_interval`):

```python
singular_values = torch.linalg.svdvals(activation_window)
cumulative_ratio = torch.cumsum(singular_values, dim=0) / singular_values.sum()
stable_rank = min(searchsorted(cumulative_ratio, 0.99) + 1, num_singular_values)

probs = singular_values / singular_values.sum()
effective_rank = exp(-(probs * log(probs)).sum())   # exp(spectral entropy)
```

| Metric | Meaning |
|---|---|
| `stable_rank` | Smallest number of singular values whose cumulative sum reaches 99% of the total — how many directions actually carry the layer's variance. |
| `stable_rank_pct` | `stable_rank` as a percentage of the layer's unit count (`100 * stable_rank / num_units`) — comparable across layers of different widths. |
| `effective_rank` | `exp(entropy(normalized singular values))` — a smoother, continuous analogue of `stable_rank`; falls as the singular-value spectrum concentrates on fewer directions. |

### ReDo dormancy score — [Sokar2023]

**Key idea:** a layer's per-unit absolute-activation EMA, divided by the
average of that EMA across the layer's own neurons. A unit whose average
activation magnitude has shrunk to near-negligible relative to its
layer-mates is "dormant" — a softer, earlier precursor to a fully dead unit.
Unlike the window metrics above, this is a continuous EMA
(`state.redo.activation_abs_ema`, decayed by `utility_decay`, never reset;
`_summarize_redo`):

```python
dormancy_score = activation_abs_ema / activation_abs_ema.mean().clamp_min(1e-12)
```

| Metric | Meaning |
|---|---|
| `redo_dormancy_score_mean` | Mean dormancy score across units in the layer (network-average score is always 1.0 in expectation; layers with more dormant units pull this below 1). |
| `redo_dormancy_score_p10` | 10th percentile of the dormancy score, i.e. 90% of the layer's neurons have an activation magnitude above this multiple of the layer average. |
| `redo_dormant_units_frac` | Fraction of units with `dormancy_score <= dormant_threshold` (default 0.1) — ReDo's actual dormant-unit classification. |
| `redo_activity_frac_mean` | Mean of a separate EMA — the fraction of forward passes on which each unit's `\|activation\|` exceeded `activity_threshold` (1e-5). A softer, continuously-decayed cousin of `active_fraction_mean` above. |

### Contribution utility — [Dohare2024] (CBP / Generate-and-Test)

**Key idea:** unlike the metrics above, which describe *symptoms*, this one
is the *mechanism* — it is the exact ranking signal Continual Backprop would
use to pick which units to reset. It measures how much a unit's output
actually moves the network's predictions: the magnitude of its outgoing
weights times how much it actually fires. Bias-corrected EMA, decayed by
`utility_decay`, never reset (`_update_activation_metrics`,
`_summarize_utility`):

```python
output_weight_magnitude = mean(|outgoing weight|, over each consumer's output units, then averaged across consumers)
instantaneous_utility = output_weight_magnitude * mean(|activation|, over the batch)
utility.ema = decay * utility.ema + (1 - decay) * instantaneous_utility
bias_corrected = utility.ema / (1 - decay ** age)     # Adam-style bias correction
```

| Metric | Meaning |
|---|---|
| `contribution_utility_mean` | Mean bias-corrected utility across units in the layer. |
| `contribution_utility_p10` | 10th percentile of utility — the layer's least-useful 10% of units sit below this value. |
| `contribution_utility_min` | The single lowest-utility unit's value — the layer's weakest link. |

### KNIFE relative update activity — [Liu2026]

**Key idea:** every metric above is activation-side. A unit can look
perfectly alive by all of them — firing normally, contributing to the
output — yet have effectively stopped *learning*: its weights are barely
moving anymore relative to their own size. KNIFE catches that gradient-side
failure mode directly, from a backward hook on the producing layer's weight
(`_update_gradient_metrics`, `_summarize_knife`/`_summarize_knife_average`):

```python
grad_norm = weight_grad.norm(p=2, dim=1)     # per output unit
weight_norm = weight.norm(p=2, dim=1)
update_activity = grad_norm / (weight_norm + rua_eps)
rua = update_activity / update_activity.mean().clamp_min(1e-12)   # "relative update activity"
```

Reported for a recent window (reset every `knife_interval` × `log_interval`
epochs) and, with a `_lifetime` suffix, as a cumulative running average that
never resets:

| Metric | Meaning |
|---|---|
| `knife_update_activity_mean` / `knife_update_activity_lifetime_mean` | Mean per-unit gradient-norm-to-weight-norm ratio across the layer. |
| `knife_rua_mean` / `knife_rua_lifetime_mean` | Mean of `update_activity` normalized by its own layer mean (RUA ≈ 1 is "typical" for the layer). |
| `knife_rua_p10` / `knife_rua_lifetime_p10` | 10th percentile of RUA — 90% of units in the layer are updating at least this fast relative to the layer average. |
| `knife_stagnant_units_frac` / `knife_stagnant_units_lifetime_frac` | Fraction of units with `RUA < stagnant_threshold` (default 0.25) — updating far slower than their layer-mates. |
| `knife_volatile_units_frac` / `knife_volatile_units_lifetime_frac` | Fraction of units with `RUA > volatile_threshold` (default 4.0) — updating unusually fast/unstably. |

### Weight magnitude (no paper — plain diagnostic)

Emitted for **every** `nn.Linear` module in the model, not just discovered
feature sites (`_add_weight_magnitude_metrics`):

| Metric | Meaning |
|---|---|
| `average_weight_magnitude` | Mean `\|weight\|` for this specific `nn.Linear` module. |

---

## Overall / Network-Wide Metrics

Computed once per manager per firing epoch, aggregating across all of that
manager's sites (`_add_network_summary_metrics`). Logged as
`plasticity/{network}/{metric}` — no per-site suffix.

### Unit-count-weighted means

The following per-site metrics are also reported network-wide as a mean
weighted by each site's unit count (`sum(value_i * num_units_i) /
sum(num_units_i)`), so wider layers contribute proportionally more:

`dead_units_frac`, `active_fraction_mean`, `dead_units_lifetime_frac`,
`active_lifetime_fraction_mean`, `redo_dormant_units_frac`,
`redo_activity_frac_mean`, `contribution_utility_mean`,
`knife_update_activity_mean`, `knife_stagnant_units_frac`,
`knife_volatile_units_frac`, `knife_update_activity_lifetime_mean`,
`knife_stagnant_units_lifetime_frac`, `knife_volatile_units_lifetime_frac`.

### Mean-of-p10 vs. global p10

For `contribution_utility_p10`, `knife_rua_p10`, and
`knife_rua_lifetime_p10`, two different network-wide numbers are logged
(`_add_mean_and_global_p10`):

| Metric | Meaning |
|---|---|
| `{metric}_mean` | Plain average of each site's own 10th-percentile value — treats every site equally regardless of width. |
| `{metric}_global` | The 10th percentile recomputed over every unit from every site pooled into one distribution — dominated by the network's widest layers. |

### Rank aggregates

For `stable_rank`, `stable_rank_pct`, and `effective_rank`
(`_add_rank_aggregates`):

| Metric | Meaning |
|---|---|
| `{metric}_mean` | Unweighted mean across sites. |
| `{metric}_final_layer` | The value from the last discovered site in the trunk — typically the layer closest to the policy/value head, and often the most diagnostic single number for representational collapse. |

### Network-wide weight magnitude

| Metric | Meaning |
|---|---|
| `average_weight_magnitude` | Mean `\|weight\|` over every `nn.Linear` in the model, weighted by each layer's parameter count (`numel`). |

---

## Config quick reference

All keys below go inside the agent config's `plasticity:` block and are
forwarded verbatim to `NetworkPlasticityManager` (validated against
`PLASTICITY_CONFIG_KEYS` in `rl_games/common/a2c_common.py` — an unknown key
raises an error naming it, rather than being silently dropped).

| Key | Default | Effect |
|---|---|---|
| `log_interval` | 10 | `summary()` calls between logged summaries |
| `rank_interval` | 50 | `summary()` calls between SVD rank calculations |
| `knife_interval` | 1 | Multiple of `log_interval` between KNIFE gradient diagnostics |
| `utility_decay` | 0.99 | EMA decay for ReDo and contribution-utility running estimates |
| `activity_threshold` | 1e-5 | Minimum `\|activation\|` counted as "active" for the ReDo activity-fraction EMA |
| `dormant_threshold` | 0.1 | ReDo dormancy-score cutoff for `redo_dormant_units_frac` |
| `stagnant_threshold` | 0.25 | KNIFE RUA cutoff for `knife_stagnant_units_frac` |
| `volatile_threshold` | 4.0 | KNIFE RUA cutoff for `knife_volatile_units_frac` |
| `rua_eps` | 1e-8 | Numerical floor in the KNIFE update-activity denominator |
| `activation_window_size` | 10000 | Max rows retained in the rolling activation buffer for dead-unit/rank stats |
| `rollout_samples_per_forward` | `None` | Caps rows sampled per rollout forward pass before adding to the activation window (`None` = no subsampling) |
| `compute_rank` | `True` | Gate for rank metrics (`stable_rank`/`effective_rank`/`stable_rank_pct`) |
| `training_only` | `True` | Restricts hook accumulation to `model.training` mode |
| `replacement_enabled` | `False` | CBP unit replacement (not yet implemented — raises if set) |

Fixed constants that are not configurable: the dead-unit threshold (`active
fraction < 0.01`), the quantile used for every `p10`/`global` metric
(`0.10`), and the SVD cumulative-energy threshold for `stable_rank`
(`0.99`).
