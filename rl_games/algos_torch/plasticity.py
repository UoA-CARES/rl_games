# rl_games/algos_torch/plasticity.py
#
# Adapted from cares_reinforcement_learning/networks/plasticity.py (see
# docs/PLASTICITY_IN_CARES_RL.md for how that reference works). This file is
# a fresh implementation, not a copy: rl_games' actor/critic heads (mu,
# sigma, value) live outside the trunk's nn.Sequential, so site discovery
# needs an explicit `output_consumers` fallback the reference doesn't have
# (see docs/PLASTICITY_STEP_1.md, section 1a).
#
# This first pass only covers site discovery (1a) - hooks, summary(), and
# step_replacement() are added in later steps.

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

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


@dataclass
class FeatureSite:
    name: str
    producer_module: nn.Linear
    hook_module: nn.Module
    consumer_modules: tuple[nn.Linear, ...]


class NetworkPlasticityManager:
    """Tracks and reports neuron-level plasticity diagnostics for a model.

    This pass only builds `self.sites` (site discovery, Step 1a). Hooks,
    summary(), and step_replacement() are added in later steps.
    """

    SUPPORTED_ACTIVATIONS = SUPPORTED_ACTIVATIONS

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        output_consumers: list[nn.Linear] | None = None,
        name: str = "network",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.output_consumers: tuple[nn.Linear, ...] = tuple(output_consumers or ())
        self.name = name

        self.sites: list[FeatureSite] = []
        self._discover_sites()

    # =========================================================================
    # Site discovery
    #
    # Walks every nn.Sequential inside the model looking for
    # Linear -> Activation -> [next Linear] triples ("feature sites"). Same
    # search as the reference implementation, with one difference: when the
    # in-Sequential search finds no consumer (the trunk's last hidden layer,
    # whose real consumers - mu/value/etc. - sit outside the Sequential),
    # fall back to the explicit `output_consumers` passed at construction
    # instead of skipping the site (see docs/PLASTICITY_STEP_1.md, 1a).
    # =========================================================================

    def _is_activation(self, module: nn.Module) -> bool:
        return isinstance(module, self.SUPPORTED_ACTIVATIONS)

    def _has_trainable_parameters(self, module: nn.Module) -> bool:
        return any(parameter.requires_grad for parameter in module.parameters())

    def _is_supported_trainable_layer(self, module: nn.Module) -> bool:
        return isinstance(module, SUPPORTED_TRAINABLE_LAYERS)

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
                    consumer_modules = self.output_consumers
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
