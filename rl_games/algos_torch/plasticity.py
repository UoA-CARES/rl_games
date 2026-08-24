# rl_games/algos_torch/plasticity.py
#
# Adapted from cares_reinforcement_learning/networks/plasticity.py (see
# docs/PLASTICITY_IN_CARES_RL.md for how that reference works). This is a
# fresh implementation, not a copy - rl_games' network shape and training
# loop differ from the reference in ways that require real structural
# changes, tracked in docs/PLASTICITY_STEP_1.md:
#
#   1a. FeatureSite.consumer_modules is a tuple, not a single module, and
#       falls back to an explicit `output_consumers` list when a site's
#       consumer isn't found inside the same nn.Sequential (rl_games' mu/
#       sigma/value heads sit outside the trunk's Sequential).
#   1b. summary() cadence copies the reference's step_count/modulo approach
#       directly - traced both codebases' actual call sites (see revised
#       docs/PLASTICITY_STEP_1.md) and summary() is only ever called once
#       per epoch in both, so there's no mismatch between "how often summary()
#       is called" and "how much real training progress that represents" to
#       correct for. No frame-based gating needed. See _summary_schedule below.
#   1c. Rollout activation subsampling - not yet added (later step).
#   1d. Site discovery skips shapes it doesn't understand (RNN/CNN/
#       normalization layers between an activation and the next Linear)
#       rather than misinterpreting them. Implemented inside _discover_sites.
#
# This pass adds hook registration, per-step metric accumulation, and
# summary() (1a + 1b + 1d). PlasticityAdam, step_replacement()/CBP reset,
# optimizer_steps (drives replacement cadence, unrelated to logging cadence),
# and 1c's rollout subsampling are separate, later steps.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterator

import torch
from torch import nn
from torch.optim import Optimizer

SUPPORTED_TRAINABLE_LAYERS = (nn.Linear,)

SUPPORTED_ACTIVATIONS = (
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.GELU,
    nn.SiLU,
    nn.Tanh,
    nn.Sigmoid,
)


class CaptureMode(Enum):
    """Controls which plasticity state the hooks are allowed to update."""

    TRAINING = auto()
    METRICS = auto()


@dataclass
class FeatureSite:
    name: str
    producer_module: nn.Linear
    hook_module: nn.Module
    consumer_modules: tuple[nn.Linear, ...]


@dataclass
class RunningAverageState:
    total: torch.Tensor
    count: torch.Tensor


@dataclass
class ActivityState:
    window: RunningAverageState
    lifetime: RunningAverageState
    behaviour_chunks: list[torch.Tensor]
    behaviour_rows: int = 0


@dataclass
class UtilityState:
    ema: torch.Tensor
    bias_corrected: torch.Tensor
    mean_feature_activation: torch.Tensor
    age: torch.Tensor


@dataclass
class RedoState:
    activation_abs_ema: torch.Tensor
    activity_fraction_ema: torch.Tensor


@dataclass
class KnifeState:
    window: RunningAverageState
    lifetime: RunningAverageState


@dataclass
class SiteState:
    utility: UtilityState
    activity: ActivityState
    redo: RedoState
    knife: KnifeState


@dataclass
class SiteSummary:
    num_units: float
    metrics: dict[str, float]
    distributions: dict[str, torch.Tensor]


class NetworkPlasticityManager:
    """Tracks neuron-level plasticity diagnostics for one sub-network.

    Two things happen at construction: site discovery (_discover_sites,
    1a/1d) and hook registration (_register_hooks). Everything else runs
    continuously during training, reading/writing the state those hooks
    feed:
      - capture_training() / capture_metrics(): context managers the
        training loop wraps around forward/backward passes so hooks only
        record during the exact spans that should count.
      - summary(): periodically returns diagnostics, at a cadence driven by
        an internal step_count incremented once per call (1b) - copies the
        reference's approach directly, since summary() is only ever called
        once per epoch in both codebases (see docs/PLASTICITY_STEP_1.md).

    optimizer_steps / on_optimizer_step() are a separate, unrelated counter
    (drives future CBP replacement cadence, not logging cadence) - left out
    of this pass.
    """

    SUPPORTED_ACTIVATIONS = SUPPORTED_ACTIVATIONS
    
# =========================================================================
    def __init__(
        self,
        model: nn.Module,  # Network whose feature sites and plasticity signals are tracked.
        optimizer: Optimizer,  # Optimizer used to read parameter updates for utility metrics.
        output_consumers: list[nn.Linear] | None = None,  # External linear heads consuming trunk features.
        name: str = "network",  # Label used to identify this network in summaries and diagnostics.
        enabled: bool = True,  # Enables or disables discovery, hooks, and metric collection.
        training_only: bool = True,  # Restricts collection to training mode when enabled.
        utility_decay: float = 0.99,  # Exponential decay applied to the running utility estimate.
        activity_threshold: float = 1e-5,  # Minimum activation magnitude counted as neuron activity.
        dormant_threshold: float = 0.1,  # Activity ratio below which a neuron is classified as dormant.
        stagnant_threshold: float = 0.25,  # Update/utilization threshold for classifying a neuron as stagnant.
        volatile_threshold: float = 4.0,  # Upper update/utilization threshold for classifying a neuron as volatile.
        rua_eps: float = 1e-8,  # Numerical floor used by relative-update calculations.
        activation_window_size: int = 10000,  # Number of recent activations retained for distribution statistics.
        rollout_samples_per_forward: int | None = None,  # Cap rows taken from a single rollout forward pass (e.g. [num_envs, hidden]) before adding to the window. None = no subsampling.
        compute_rank: bool = True,  # Whether to compute rank-based feature diagnostics.
        log_interval: int = 10,  # summary() calls (= epochs) between logged summaries. Alarm 1
        rank_interval: int = 50,  # summary() calls (= epochs) between rank calculations. Alarm 2
        knife_interval: int = 1,  # Multiple of log_interval between knife diagnostics. Alarm 3
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        # External heads are used when a feature site is the final layer of a
        # trunk (for example, policy and value heads living outside the
        # Sequential).  Keep only Linear modules with a valid feature
        # interface; this makes the fallback useful for heterogeneous models
        # and prevents an unrelated head from corrupting utility metrics.
        self.output_consumers: tuple[nn.Linear, ...] = tuple(
            consumer
            for consumer in (output_consumers or ())
            if isinstance(consumer, nn.Linear)
        )
        self.name = name

        self.enabled = enabled
        self.training_only = training_only
        self.utility_decay = utility_decay
        self.activity_threshold = activity_threshold
        self.dormant_threshold = dormant_threshold
        self.stagnant_threshold = stagnant_threshold
        self.volatile_threshold = volatile_threshold
        self.rua_eps = rua_eps
        self.activation_window_size = activation_window_size
        self.rollout_samples_per_forward = rollout_samples_per_forward
        self.compute_rank = compute_rank
        self.log_interval = log_interval
        self.rank_interval = rank_interval
        self.knife_interval = knife_interval

        # 1b: matches the reference directly - a single counter incremented
        # once per summary() call (once per epoch, in both codebases), not
        # a frame-based counter. See class docstring and docs/PLASTICITY_STEP_1.md.
        self.step_count = 0

        self.sites: list[FeatureSite] = []
        self.handles: list[Any] = []
        self.grad_handles: list[Any] = []
        self.site_states: dict[str, SiteState] = {}
        self.last_summary: dict[str, float] = {}

        # Hooks are inactive unless a capture context is active. A stack
        # safely restores the previous mode if capture contexts nest, and
        # keeps any forward pass not explicitly wrapped by the training
        # loop (eval passes, GAE bootstrapping, etc.) from silently
        # polluting plasticity statistics.
        self._capture_modes: list[CaptureMode] = []

        if self.enabled:
            self._discover_sites()
            self._register_hooks()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

        for handle in self.grad_handles:
            handle.remove()
        self.grad_handles.clear()
        
# =========================================================================

    @property
    def _active_capture_mode(self) -> CaptureMode | None:
        return self._capture_modes[-1] if self._capture_modes else None

    @contextmanager
    def _capture(self, mode: CaptureMode) -> Iterator[None]:
        self._capture_modes.append(mode)
        try:
            yield
        finally:
            popped_mode = self._capture_modes.pop()
            if popped_mode is not mode:
                raise RuntimeError("Plasticity capture contexts exited out of order.")

    def capture_training(self):
        """Capture training activations and gradients inside this context."""
        return self._capture(CaptureMode.TRAINING)

    def capture_metrics(self):
        """Capture rollout activations for rank/activity diagnostics."""
        return self._capture(CaptureMode.METRICS)

    # =========================================================================
    # Site discovery & hook registration (1a, 1d)
    # =========================================================================

    def _is_activation(self, module: nn.Module) -> bool:
        return isinstance(module, self.SUPPORTED_ACTIVATIONS)

    def _has_trainable_parameters(self, module: nn.Module) -> bool:
        return any(parameter.requires_grad for parameter in module.parameters())

    def _is_supported_trainable_layer(self, module: nn.Module) -> bool:
        return isinstance(module, SUPPORTED_TRAINABLE_LAYERS)

# This method is to check where to discover the sites in the model. It looks for nn.Sequential modules and checks for
# supported trainable layers followed by supported activation functions. If found, it records the site information for later
# use in plasticity tracking.
    def _discover_sites(self) -> None:
        for seq_name, seq in self.model.named_modules():
            if not isinstance(seq, nn.Sequential):
                continue

            children = list(seq.named_children())

            for idx, (child_name, child) in enumerate(children):
                if not self._is_supported_trainable_layer(child):
                    continue

                if idx + 1 >= len(children):
                    continue

                _, possible_activation = children[idx + 1]
                if not self._is_activation(possible_activation):
                    continue

                consumer_module: nn.Linear | None = None
                unsupported_between = False

                # 1d: anything else with trainable parameters between the
                # activation and the next Linear (RNN, LayerNorm, ...) means
                # this site's shape isn't one we understand yet - skip it
                # rather than misinterpret it as Linear -> Linear.
                for _, later_module in children[idx + 2 :]:
                    if self._is_supported_trainable_layer(later_module):
                        consumer_module = later_module
                        break

                    if self._has_trainable_parameters(later_module):
                        unsupported_between = True
                        break

                if unsupported_between:
                    continue

                if consumer_module is not None:
                    consumer_modules: tuple[nn.Linear, ...] = (consumer_module,)
                else:
                    # 1a: nothing left in this Sequential - fall back to the
                    # explicit external consumers (e.g. mu/value heads).
                    consumer_modules = tuple(
                        consumer
                        for consumer in self.output_consumers
                        if consumer.in_features == child.out_features
                    )
                    if not consumer_modules:
                        continue

                site_name = f"{seq_name}.{child_name}" if seq_name else child_name

                self.sites.append(
                    FeatureSite(
                        name=site_name,
                        producer_module=child,
                        hook_module=possible_activation,
                        consumer_modules=consumer_modules,
                    )
                )

## After the site has been discovered this method registers the forward and backward hooks for each site. 
# The forward hook captures the activations during the forward pass, while the backward hook captures the gradients during the backward pass.
# These hooks are essential for tracking the plasticity metrics of the network.
    def _register_hooks(self) -> None:
        for site in self.sites:
            forward_handle = site.hook_module.register_forward_hook(
                self._make_forward_hook(site)
            )
            self.handles.append(forward_handle)

            grad_handle = site.producer_module.weight.register_hook(
                self._make_weight_grad_hook(site)
            )
            self.grad_handles.append(grad_handle)

## This method is used to calculate the dead neurons, dormant neurons, etc
    def _make_forward_hook(self, site: FeatureSite):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not self.enabled or self._active_capture_mode is None:
                return

            if not torch.is_tensor(output):
                return

            activation = output.detach()
            if activation.ndim != 2:
                return

            if self._active_capture_mode is CaptureMode.METRICS:
                self._record_activation_window(site, activation)
                return

            if self._active_capture_mode is CaptureMode.TRAINING:
                if self.training_only and not self.model.training:
                    return
                self._update_activation_metrics(site, activation)

        return hook

# This method is used to calulate the gradient during backward pass. It checks if the plasticity manager is enabled and 
# if the current capture mode is TRAINING. If so, it updates the gradient metrics for the given site.
    def _make_weight_grad_hook(self, site: FeatureSite):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if (
                not self.enabled
                or self._active_capture_mode is not CaptureMode.TRAINING
            ):
                return grad

            if self.training_only and not self.model.training:
                return grad

            if not torch.is_tensor(grad):
                return grad

            self._update_gradient_metrics(site, grad.detach())
            return grad

        return hook

    # =========================================================================
    # Per-step metric recording (called from the hooks above)
    # =========================================================================

    @torch.no_grad()
    def _ensure_site_state(
        self,
        site: FeatureSite,
        num_units: int,
        device: torch.device,
    ) -> SiteState:
        existing = self.site_states.get(site.name)
        if existing is not None:
            return existing

        def zeros() -> torch.Tensor:
            return torch.zeros(num_units, device=device)

        state = SiteState(
            utility=UtilityState(
                ema=zeros(),
                bias_corrected=zeros(),
                mean_feature_activation=zeros(),
                age=zeros(),
            ),
            activity=ActivityState(
                window=RunningAverageState(total=zeros(), count=zeros()),
                lifetime=RunningAverageState(total=zeros(), count=zeros()),
                behaviour_chunks=[],
            ),
            redo=RedoState(
                activation_abs_ema=zeros(),
                activity_fraction_ema=zeros(),
            ),
            knife=KnifeState(
                window=RunningAverageState(total=zeros(), count=zeros()),
                lifetime=RunningAverageState(total=zeros(), count=zeros()),
            ),
        )
        self.site_states[site.name] = state
        return state

    @torch.no_grad()
    def _record_activation_window(
        self,
        site: FeatureSite,
        activation: torch.Tensor,
    ) -> None:
        state = self._ensure_site_state(site, activation.shape[1], activation.device)

        max_rows = int(self.activation_window_size)
        if max_rows <= 0:
            return

        limit = self.rollout_samples_per_forward
        if limit is not None and activation.shape[0] > limit:
            idx = torch.randperm(activation.shape[0], device=activation.device)[:limit]
            activation = activation[idx]

        chunk = activation.detach().float().cpu()
        state.activity.behaviour_chunks.append(chunk)
        state.activity.behaviour_rows += int(chunk.shape[0])

        while (
            state.activity.behaviour_chunks and state.activity.behaviour_rows > max_rows
        ):
            removed = state.activity.behaviour_chunks.pop(0)
            state.activity.behaviour_rows -= int(removed.shape[0])

    @torch.no_grad()
    def _activation_window_tensor(self, state: SiteState) -> torch.Tensor | None:
        chunks = state.activity.behaviour_chunks
        if not chunks:
            return None

        window = torch.cat(chunks, dim=0)
        max_rows = int(self.activation_window_size)
        if max_rows > 0 and window.shape[0] > max_rows:
            window = window[-max_rows:]
        return window

    @torch.no_grad()
    def _consumer_utility_weight(self, site: FeatureSite) -> torch.Tensor:
        # |output weight| for each producer unit, averaged over each
        # consumer's own output units, then averaged across consumers.
        # rl_games' last trunk layer has two consumers (mu, value) with
        # different output widths, so each consumer is reduced to a single
        # per-unit vector first, and those vectors are then combined - no
        # single consumer is allowed to dominate just because it has more
        # output units (e.g. mu's action dimension) than another (value's 1).
        per_consumer = [
            consumer.weight.detach().abs().mean(dim=0)
            for consumer in site.consumer_modules
        ]
        return torch.stack(per_consumer, dim=0).mean(dim=0)

    @torch.no_grad()
    def _update_activation_metrics(
        self,
        site: FeatureSite,
        activation: torch.Tensor,
    ) -> None:
        state = self._ensure_site_state(site, activation.shape[1], activation.device)
        decay = self.utility_decay

        state.utility.age.add_(1.0)

        bias_correction = 1.0 - torch.pow(
            torch.tensor(decay, device=activation.device), state.utility.age
        )
        bias_correction.clamp_min_(1e-12)

        activation_abs_mean = activation.abs().mean(dim=0)

        batch_active_fraction = (activation > 0.0).float().mean(dim=0)
        state.activity.window.total.add_(batch_active_fraction)
        state.activity.window.count.add_(1.0)
        state.activity.lifetime.total.add_(batch_active_fraction)
        state.activity.lifetime.count.add_(1.0)

        state.redo.activation_abs_ema.mul_(decay).add_(
            (1.0 - decay) * activation_abs_mean
        )
        batch_activity = (
            (activation.abs() > self.activity_threshold).float().mean(dim=0)
        )
        state.redo.activity_fraction_ema.mul_(decay).add_(
            (1.0 - decay) * batch_activity
        )

        state.utility.mean_feature_activation.mul_(decay).add_(
            (1.0 - decay) * activation.mean(dim=0)
        )

        output_weight_magnitude = self._consumer_utility_weight(site)
        instantaneous_utility = output_weight_magnitude * activation_abs_mean

        state.utility.ema.mul_(decay).add_((1.0 - decay) * instantaneous_utility)
        state.utility.bias_corrected.copy_(state.utility.ema / bias_correction)

    @torch.no_grad()
    def _update_gradient_metrics(
        self,
        site: FeatureSite,
        weight_grad: torch.Tensor,
    ) -> None:
        if weight_grad.ndim != 2:
            return

        state = self._ensure_site_state(site, weight_grad.shape[0], weight_grad.device)
        weight = site.producer_module.weight.detach()

        grad_norm = weight_grad.norm(p=2, dim=1)
        weight_norm = weight.norm(p=2, dim=1)
        update_activity = grad_norm / (weight_norm + self.rua_eps)

        state.knife.window.total.add_(update_activity)
        state.knife.window.count.add_(1.0)
        state.knife.lifetime.total.add_(update_activity)
        state.knife.lifetime.count.add_(1.0)

    # =========================================================================
    # summary() - public entry point. Diagnostics at a step_count-driven
    # cadence, incremented once per call - same as the reference (1b).
    # =========================================================================

    @torch.no_grad()
    def summary(
        self,
        prefix: str | None = None,
        force: bool = False,
    ) -> dict[str, float]:
        """Return diagnostics if a logging/ranking interval was reached.

        No frame argument: cadence is judged purely by how many times
        summary() itself has been called (self.step_count), matching the
        reference exactly. summary() is only ever called once per epoch in
        both codebases (see docs/PLASTICITY_STEP_1.md), so this already
        represents real training progress - the caller (write_stats) tags
        the logged scalars with the real frame number separately, at the
        add_scalar call site, purely as an x-axis label.
        """
        if not self.enabled:
            return {}

        self.step_count += 1
        should_log, should_rank, should_knife = self._summary_schedule(force)
        if not should_log and not should_rank:
            return {}

        prefix = prefix or self.name
        info: dict[str, float] = {}
        site_summaries: list[SiteSummary] = []

        for site in self.sites:
            state = self.site_states.get(site.name)
            if state is None:
                continue

            summary = self._summarize_site(
                site=site,
                state=state,
                should_rank=should_rank,
                should_knife=should_knife,
            )
            site_summaries.append(summary)

            clean_name = site.name.replace(".", "_")
            for metric_name, value in summary.metrics.items():
                info[f"{prefix}/{clean_name}/{metric_name}"] = value

        self._add_network_summary_metrics(info, prefix, site_summaries)
        self._add_weight_magnitude_metrics(info, prefix)

        self._reset_summary_windows(should_log, should_knife)
        self.last_summary = info
        return info

    def _summary_schedule(self, force: bool) -> tuple[bool, bool, bool]:
        # Plain modulo, matching the reference exactly - safe here because
        # self.step_count only ever advances by exactly 1 per call, so it
        # can never skip past a multiple the way a raw frame count could.
        should_log = force or self.step_count % self.log_interval == 0
        should_rank = self.compute_rank and (
            force or self.step_count % self.rank_interval == 0
        )

        # KNIFE is gated as a multiple of log occurrences (how many times
        # should_log has fired), not an independent step_count check - so
        # its accumulation window always aligns with a should_log reset.
        log_count = self.step_count // self.log_interval
        should_knife = force or (should_log and log_count % self.knife_interval == 0)

        return should_log, should_rank, should_knife

    @torch.no_grad()
    def _summarize_site(
        self,
        site: FeatureSite,
        state: SiteState,
        should_rank: bool,
        should_knife: bool,
    ) -> SiteSummary:
        result = SiteSummary(
            num_units=float(site.producer_module.out_features),
            metrics={},
            distributions={},
        )

        self._summarize_recent_activity(result, state, should_rank)
        self._summarize_lifetime_activity(result, state)
        self._summarize_redo(result, state)
        self._summarize_utility(result, state)

        if should_knife:
            self._summarize_knife(result, state)

        return result

    @torch.no_grad()
    def _summarize_recent_activity(
        self,
        result: SiteSummary,
        state: SiteState,
        should_rank: bool,
    ) -> None:
        # dead_units_frac / rank metrics reproduce the Fig. 2d/4b
        # diagnostics from Dohare et al. 2024 ("Loss of Plasticity in Deep
        # Continual Learning") - see PLASTICITY_IN_CARES_RL.md for the full
        # citation list this mirrors.
        activity_window = self._activation_window_tensor(state)
        if activity_window is None or activity_window.shape[0] == 0:
            return

        active_fraction = (activity_window > 0.0).float().mean(dim=0)
        result.metrics.update(
            {
                "dead_units_frac": ((active_fraction < 0.01).float().mean().item()),
                "active_fraction_mean": active_fraction.mean().item(),
                "active_fraction_p10": torch.quantile(active_fraction, 0.10).item(),
                "activity_window_size": float(activity_window.shape[0]),
            }
        )

        if should_rank:
            result.metrics.update(
                self._rank_metrics(activity_window, num_units=result.num_units)
            )

    @torch.no_grad()
    def _rank_metrics(
        self, activation: torch.Tensor, num_units: float | None = None
    ) -> dict[str, float]:
        if activation.shape[0] < 2:
            return {}

        x = activation.float()

        try:
            singular_values = torch.linalg.svdvals(x)
        except RuntimeError:
            return {}

        if singular_values.numel() == 0:
            return {}

        singular_sum = singular_values.sum()
        if singular_sum <= 0:
            metrics = {"stable_rank": 0.0, "effective_rank": 0.0}
            if num_units:
                metrics["stable_rank_pct"] = 0.0
            return metrics

        cumulative_ratio = torch.cumsum(singular_values, dim=0) / singular_sum
        stable_rank = float(torch.searchsorted(cumulative_ratio, 0.99).item() + 1)
        stable_rank = min(stable_rank, float(singular_values.numel()))

        probs = singular_values / singular_sum.clamp_min(1e-12)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
        effective_rank = torch.exp(entropy).item()

        metrics = {"stable_rank": stable_rank, "effective_rank": effective_rank}
        if num_units:
            metrics["stable_rank_pct"] = 100.0 * stable_rank / num_units

        return metrics

    @torch.no_grad()
    def _summarize_lifetime_activity(
        self,
        result: SiteSummary,
        state: SiteState,
    ) -> None:
        active_fraction = (
            state.activity.lifetime.total / state.activity.lifetime.count.clamp_min(1.0)
        )
        result.metrics.update(
            {
                "dead_units_lifetime_frac": (
                    (active_fraction < 0.01).float().mean().item()
                ),
                "active_lifetime_fraction_mean": active_fraction.mean().item(),
            }
        )

    @torch.no_grad()
    def _summarize_redo(
        self,
        result: SiteSummary,
        state: SiteState,
    ) -> None:
        activation_abs = state.redo.activation_abs_ema
        dormancy_score = activation_abs / activation_abs.mean().clamp_min(1e-12)
        result.metrics.update(
            {
                "redo_dormant_units_frac": (
                    (dormancy_score <= self.dormant_threshold).float().mean().item()
                ),
                "redo_dormancy_score_mean": dormancy_score.mean().item(),
                "redo_dormancy_score_p10": torch.quantile(dormancy_score, 0.10).item(),
                "redo_activity_frac_mean": (
                    state.redo.activity_fraction_ema.mean().item()
                ),
            }
        )

    @torch.no_grad()
    def _summarize_utility(
        self,
        result: SiteSummary,
        state: SiteState,
    ) -> None:
        utility = state.utility.bias_corrected
        result.metrics.update(
            {
                "contribution_utility_mean": utility.mean().item(),
                "contribution_utility_p10": torch.quantile(utility, 0.10).item(),
                "contribution_utility_min": utility.min().item(),
            }
        )
        result.distributions["contribution_utility"] = utility.detach().flatten()

    @torch.no_grad()
    def _summarize_knife(
        self,
        result: SiteSummary,
        state: SiteState,
    ) -> None:
        self._summarize_knife_average(state.knife.window, result, lifetime=False)
        self._summarize_knife_average(state.knife.lifetime, result, lifetime=True)

    @torch.no_grad()
    def _summarize_knife_average(
        self,
        average: RunningAverageState,
        result: SiteSummary,
        lifetime: bool,
    ) -> None:
        update_activity = average.total / average.count.clamp_min(1.0)
        if not torch.any(update_activity > 0):
            return

        rua = update_activity / update_activity.mean().clamp_min(1e-12)
        suffix = "_lifetime" if lifetime else ""
        result.metrics.update(
            {
                f"knife_update_activity{suffix}_mean": (update_activity.mean().item()),
                f"knife_rua{suffix}_mean": rua.mean().item(),
                f"knife_rua{suffix}_p10": torch.quantile(rua, 0.10).item(),
                f"knife_stagnant_units{suffix}_frac": (
                    (rua < self.stagnant_threshold).float().mean().item()
                ),
                f"knife_volatile_units{suffix}_frac": (
                    (rua > self.volatile_threshold).float().mean().item()
                ),
            }
        )
        result.distributions[f"knife_rua{suffix}"] = rua.detach().flatten()

    @torch.no_grad()
    def _add_network_summary_metrics(
        self,
        info: dict[str, float],
        prefix: str,
        site_summaries: list[SiteSummary],
    ) -> None:
        if not site_summaries:
            return

        weighted_metrics = (
            "dead_units_frac",
            "active_fraction_mean",
            "dead_units_lifetime_frac",
            "active_lifetime_fraction_mean",
            "redo_dormant_units_frac",
            "redo_activity_frac_mean",
            "contribution_utility_mean",
            "knife_update_activity_mean",
            "knife_stagnant_units_frac",
            "knife_volatile_units_frac",
            "knife_update_activity_lifetime_mean",
            "knife_stagnant_units_lifetime_frac",
            "knife_volatile_units_lifetime_frac",
        )
        for metric_name in weighted_metrics:
            value = self._weighted_site_mean(site_summaries, metric_name)
            if value is not None:
                info[f"{prefix}/{metric_name}"] = value

        self._add_mean_and_global_p10(
            info, prefix, site_summaries,
            metric_name="contribution_utility_p10",
            distribution_name="contribution_utility",
        )
        self._add_mean_and_global_p10(
            info, prefix, site_summaries,
            metric_name="knife_rua_p10",
            distribution_name="knife_rua",
        )
        self._add_mean_and_global_p10(
            info, prefix, site_summaries,
            metric_name="knife_rua_lifetime_p10",
            distribution_name="knife_rua_lifetime",
        )
        self._add_rank_aggregates(info, prefix, site_summaries)

    @staticmethod
    def _weighted_site_mean(
        site_summaries: list[SiteSummary],
        metric_name: str,
    ) -> float | None:
        total_units = sum(summary.num_units for summary in site_summaries)
        if total_units <= 0:
            return None

        weighted_sum = 0.0
        found_value = False
        for summary in site_summaries:
            value = summary.metrics.get(metric_name)
            if value is None:
                continue
            weighted_sum += value * summary.num_units
            found_value = True

        return weighted_sum / total_units if found_value else None

    @staticmethod
    def _add_mean_and_global_p10(
        info: dict[str, float],
        prefix: str,
        site_summaries: list[SiteSummary],
        metric_name: str,
        distribution_name: str,
    ) -> None:
        metric_values = [
            summary.metrics[metric_name]
            for summary in site_summaries
            if metric_name in summary.metrics
        ]
        distributions = [
            summary.distributions[distribution_name]
            for summary in site_summaries
            if distribution_name in summary.distributions
        ]
        if not metric_values or not distributions:
            return

        info[f"{prefix}/{metric_name}_mean"] = sum(metric_values) / len(metric_values)
        info[f"{prefix}/{metric_name}_global"] = torch.quantile(
            torch.cat(distributions), 0.10
        ).item()

    @staticmethod
    def _add_rank_aggregates(
        info: dict[str, float],
        prefix: str,
        site_summaries: list[SiteSummary],
    ) -> None:
        for metric_name in ("stable_rank", "stable_rank_pct", "effective_rank"):
            values = [
                summary.metrics[metric_name]
                for summary in site_summaries
                if metric_name in summary.metrics
            ]
            if values:
                info[f"{prefix}/{metric_name}_mean"] = sum(values) / len(values)
                info[f"{prefix}/{metric_name}_final_layer"] = values[-1]

    @torch.no_grad()
    def _add_weight_magnitude_metrics(
        self,
        info: dict[str, float],
        prefix: str,
    ) -> None:
        total_abs_weight = 0.0
        total_weight_count = 0.0

        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            clean_name = module_name.replace(".", "_") if module_name else "linear"
            weight_abs_mean = module.weight.detach().abs().mean().item()
            weight_count = float(module.weight.numel())

            info[f"{prefix}/{clean_name}/average_weight_magnitude"] = weight_abs_mean

            total_abs_weight += weight_abs_mean * weight_count
            total_weight_count += weight_count

        if total_weight_count > 0:
            info[f"{prefix}/average_weight_magnitude"] = (
                total_abs_weight / total_weight_count
            )

    @torch.no_grad()
    def _reset_summary_windows(
        self,
        should_log: bool,
        should_knife: bool,
    ) -> None:
        if not self.model.training:
            return

        for state in self.site_states.values():
            if should_log:
                state.activity.window.total.zero_()
                state.activity.window.count.zero_()
                state.activity.behaviour_chunks.clear()
                state.activity.behaviour_rows = 0

            if should_knife:
                state.knife.window.total.zero_()
                state.knife.window.count.zero_()