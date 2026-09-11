"""Warm start (transfer) vs checkpoint resume.

`restore` / `set_full_state_weights` bring back everything an interrupted run
needs. `load_warmstart` / `set_warmstart_weights` bring back only what a
transfer onto a *different reward function* should keep. These tests pin both:
the warm-start subsets, and - just as importantly - that resume still restores
the full state it always did.

No agent is constructed. Every method under test reads a handful of attributes
off `self`, so a stand-in carrying those attributes (with a real nn.Module and a
real optimizer, so the state dicts are genuine) exercises the real code without
an environment, a network builder, or a GPU. Same approach as
tests/test_plasticity.py.
"""

import torch
from torch import nn

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.common.a2c_common import A2CBase, WARM_START_CONFIG_KEYS
from rl_games.torch_runner import _restore


OBS_SIZE = 4
CONFIG_LR = 3e-4
CONFIG_ENTROPY_COEF = 0.01
# A2CBase.__init__ sets last_mean_rewards to this; a warm start must leave it there.
NO_REWARD_YET = -100500


class _Model(nn.Module):
    """The parts of a real BaseModelNetwork these code paths touch.

    The normalizers are submodules, exactly as models.py attaches them - which
    is the whole reason they ride inside `state_dict()['model']` and cannot be
    dropped from a checkpoint by omitting a key.
    """

    def __init__(self, dict_obs=False):
        super().__init__()
        self.a2c_network = nn.Linear(OBS_SIZE, 2)
        if dict_obs:
            self.running_mean_std = RunningMeanStdObs({'a': (OBS_SIZE,), 'b': (2,)})
        else:
            self.running_mean_std = RunningMeanStd((OBS_SIZE,))
        self.value_mean_std = RunningMeanStd((1,))


class _Scaler:
    """Stand-in for the AMP GradScaler.

    A real GradScaler is disabled without CUDA and then reports an empty state
    dict, which would make "did the scaler come back?" unfalsifiable. This one
    always carries observable state and records whether it was loaded into.
    """

    def __init__(self, scale=1.0):
        self.scale = scale
        self.loaded = False

    def state_dict(self):
        return {'scale': self.scale}

    def load_state_dict(self, state):
        self.scale = state['scale']
        self.loaded = True


class _Scheduler:
    """Records whether the lr schedule was advanced, and always moves the lr.

    A real adaptive scheduler with KL == 0 raises the lr, which is exactly the
    behaviour the warmup must suppress; moving it unconditionally makes
    "was update() reached" observable from last_lr alone.
    """

    def __init__(self):
        self.calls = 0

    def update(self, lr, entropy_coef, epoch, frames, kl):
        self.calls += 1
        return lr * 2.0, entropy_coef


class _Agent:
    """Stand-in `self` for the unbound A2CBase methods under test."""

    def __init__(self, dict_obs=False, mixed_precision=False, **warm_start):
        self.model = _Model(dict_obs=dict_obs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG_LR)

        self.epoch_num = 0
        self.frame = 0
        self.last_lr = CONFIG_LR
        self.entropy_coef = CONFIG_ENTROPY_COEF
        self.last_mean_rewards = NO_REWARD_YET

        self.normalize_input = True
        self.normalize_value = True
        self.normalize_rms_advantage = False
        # The alias a2c_continuous/a2c_discrete set in __init__.
        self.value_mean_std = self.model.value_mean_std

        self.mixed_precision = mixed_precision
        self.scaler = _Scaler()
        self.has_central_value = False
        self.vec_env = None
        self.plasticity_managers = []
        self.multi_gpu = False

        # Critic warmup state, as A2CBase.__init__ initialises it.
        self.critic_warmup_epoch_count = warm_start.get('critic_warmup_epoch_count', 0)
        self.critic_warmup_until_epoch = None
        self.critic_warmup_active = False
        self._critic_warmup_saved_requires_grad = []
        self.has_value_loss = True
        self.scheduler = _Scheduler()

        self.warm_start_reset_optimizer = warm_start.get('reset_optimizer', True)
        self.warm_start_reset_lr_schedule = warm_start.get('reset_lr_schedule', True)
        self.warm_start_reset_obs_normalizer = warm_start.get('reset_obs_normalizer', False)
        self.warm_start_reset_value_normalizer = warm_start.get('reset_value_normalizer', True)

    # The methods under test, called unbound so the real implementations run.
    save_state = A2CBase.get_full_state_weights
    load_resume = A2CBase.set_full_state_weights
    load_warm = A2CBase.set_warmstart_weights
    set_weights = A2CBase.set_weights
    set_stats_weights = A2CBase.set_stats_weights
    get_weights = A2CBase.get_weights
    get_stats_weights = A2CBase.get_stats_weights
    update_lr = A2CBase.update_lr
    _restore_plasticity_state = A2CBase._restore_plasticity_state
    _critic_warmup_frozen_parameters = A2CBase._critic_warmup_frozen_parameters
    _set_critic_warmup = A2CBase._set_critic_warmup
    _update_critic_warmup_state = A2CBase._update_critic_warmup_state
    _maybe_update_schedule = A2CBase._maybe_update_schedule


def _feed(rms, value):
    """Push statistics into a RunningMeanStd so they are no longer the defaults."""
    rms.train()
    rms(torch.full((8, rms.insize[0] if isinstance(rms.insize, tuple) else rms.insize), value))
    rms.eval()


def _trained_source(**kwargs):
    """An agent that looks like it has been through some training."""
    agent = _Agent(**kwargs)

    # Distinct weights, so a transfer is visible.
    with torch.no_grad():
        agent.model.a2c_network.weight.fill_(0.5)
        agent.model.a2c_network.bias.fill_(-0.25)

    # Real optimizer moments, so "was the optimizer state carried over" is a
    # question about actual tensors rather than an empty dict.
    loss = agent.model.a2c_network(torch.ones(1, OBS_SIZE)).sum()
    loss.backward()
    agent.optimizer.step()
    agent.optimizer.zero_grad()

    # Distinct, non-initial normalizer statistics - and different from each
    # other, so a test cannot pass by resetting or keeping both.
    _feed(agent.model.value_mean_std, 7.0)
    if isinstance(agent.model.running_mean_std, RunningMeanStd):
        _feed(agent.model.running_mean_std, 3.0)
    else:
        for sub in agent.model.running_mean_std.running_mean_std.values():
            _feed(sub, 3.0)

    agent.scaler.scale = 512.0

    agent.epoch_num = 120
    agent.frame = 48000
    agent.last_mean_rewards = 42.5
    agent.last_lr = 9e-5           # an adaptive schedule moved it away from config
    agent.entropy_coef = 0.004
    A2CBase.update_lr(agent, agent.last_lr)
    return agent


def _checkpoint(agent, tmp_path, name='ckpt'):
    """Round-trip through the real save/load, not just an in-memory dict."""
    path = str(tmp_path / name)
    torch_ext.save_checkpoint(path, A2CBase.get_full_state_weights(agent))
    return torch_ext.load_checkpoint(path + '.pth')


def _stats(rms):
    return (rms.running_mean.clone(), rms.running_var.clone(), rms.count.clone())


def _is_initial(rms):
    mean, var, count = _stats(rms)
    return (bool(torch.all(mean == 0.0))
            and bool(torch.all(var == 1.0))
            and bool(count == 1.0))


# --------------------------------------------------------------------------- #
# 1. Saving                                                                    #
# --------------------------------------------------------------------------- #

def test_checkpoint_round_trip_carries_the_expected_keys(tmp_path):
    checkpoint = _checkpoint(_trained_source(), tmp_path)

    assert set(checkpoint) == {
        'model', 'epoch', 'optimizer', 'frame', 'last_mean_rewards',
        'last_lr', 'entropy_coef',
    }
    # The normalizer statistics are NOT top-level keys - they ride inside
    # 'model' as buffers. Every reset/keep decision below follows from this.
    assert 'running_mean_std' not in checkpoint
    assert 'reward_mean_std' not in checkpoint
    assert 'running_mean_std.running_mean' in checkpoint['model']
    assert 'value_mean_std.running_mean' in checkpoint['model']


# --------------------------------------------------------------------------- #
# 2. Resume - the regression guard. None of this may change.                    #
# --------------------------------------------------------------------------- #

def test_resume_restores_the_full_state(tmp_path):
    source = _trained_source()
    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent()
    A2CBase.set_full_state_weights(target, checkpoint)

    assert torch.equal(target.model.a2c_network.weight, source.model.a2c_network.weight)
    assert target.epoch_num == 120
    assert target.frame == 48000
    assert target.last_mean_rewards == 42.5
    assert target.last_lr == source.last_lr
    assert target.entropy_coef == source.entropy_coef

    # Optimizer moments and the lr that came back with them.
    state = target.optimizer.state_dict()['state']
    assert state and all('exp_avg' in entry for entry in state.values())
    assert target.optimizer.param_groups[0]['lr'] == source.last_lr

    # Both normalizers continue from where the source left them.
    for name in ('running_mean_std', 'value_mean_std'):
        for restored, original in zip(_stats(getattr(target.model, name)),
                                      _stats(getattr(source.model, name))):
            assert torch.equal(restored, original)


# --------------------------------------------------------------------------- #
# 3-6. Warm start: optimizer and lr schedule                                    #
# --------------------------------------------------------------------------- #

def test_warm_start_keeps_weights_and_counters_but_not_the_best_score(tmp_path):
    source = _trained_source()
    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent(reset_optimizer=False, reset_lr_schedule=False)
    A2CBase.set_warmstart_weights(target, checkpoint)

    assert torch.equal(target.model.a2c_network.weight, source.model.a2c_network.weight)
    assert target.epoch_num == 120
    assert target.frame == 48000
    assert target.last_lr == source.last_lr
    assert target.entropy_coef == source.entropy_coef

    state = target.optimizer.state_dict()['state']
    assert state and all('exp_avg' in entry for entry in state.values())
    assert target.optimizer.param_groups[0]['lr'] == source.last_lr

    # The one thing a transfer must never inherit: a best-ever score earned
    # under the previous reward function would gate every save of this run.
    assert target.last_mean_rewards == NO_REWARD_YET


def test_warm_start_reset_optimizer_drops_the_moments_and_the_scaler(tmp_path):
    source = _trained_source(mixed_precision=True)
    checkpoint = _checkpoint(source, tmp_path)
    assert checkpoint['scaler'] == {'scale': 512.0}   # there was something to drop

    target = _Agent(mixed_precision=True, reset_optimizer=True)
    A2CBase.set_warmstart_weights(target, checkpoint)

    assert target.optimizer.state_dict()['state'] == {}
    # The GradScaler's loss scale is optimizer-side state and goes with it.
    assert target.scaler.loaded is False
    assert target.scaler.scale == 1.0
    # Weights still transferred.
    assert torch.allclose(target.model.a2c_network.weight,
                          torch.full_like(target.model.a2c_network.weight, 0.5),
                          atol=1e-2)


def test_warm_start_keeping_the_optimizer_keeps_the_scaler(tmp_path):
    checkpoint = _checkpoint(_trained_source(mixed_precision=True), tmp_path)

    target = _Agent(mixed_precision=True, reset_optimizer=False)
    A2CBase.set_warmstart_weights(target, checkpoint)

    assert target.scaler.loaded is True
    assert target.scaler.scale == 512.0


def test_warm_start_reset_lr_schedule_overrides_the_optimizers_own_lr(tmp_path):
    """The trap: the optimizer state carries the checkpoint's lr in param_groups.

    Keeping the optimizer while resetting the schedule has to push the config lr
    back on, or reset_lr_schedule silently does nothing.
    """
    source = _trained_source()
    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent(reset_optimizer=False, reset_lr_schedule=True)
    A2CBase.set_warmstart_weights(target, checkpoint)

    assert target.last_lr == CONFIG_LR
    assert target.entropy_coef == CONFIG_ENTROPY_COEF
    assert target.optimizer.param_groups[0]['lr'] == CONFIG_LR
    # ... while the moments it was told to keep are still there.
    assert target.optimizer.state_dict()['state']


def test_warm_start_reset_optimizer_still_applies_the_restored_lr(tmp_path):
    source = _trained_source()
    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent(reset_optimizer=True, reset_lr_schedule=False)
    A2CBase.set_warmstart_weights(target, checkpoint)

    assert target.last_lr == source.last_lr
    assert target.entropy_coef == source.entropy_coef
    # A fresh optimizer starts at the config lr, so the restored one has to be
    # pushed onto it explicitly.
    assert target.optimizer.param_groups[0]['lr'] == source.last_lr


# --------------------------------------------------------------------------- #
# 7-9. Warm start: the normalizers                                              #
# --------------------------------------------------------------------------- #

def test_warm_start_defaults_keep_obs_stats_and_reset_value_stats(tmp_path):
    """The default split: the env didn't change, the reward did.

    Both sets of statistics arrive inside weights['model'], so an implementation
    that tried to reset by omitting a checkpoint key would fail here.
    """
    source = _trained_source()
    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent()  # defaults: obs kept, value reset
    A2CBase.set_warmstart_weights(target, checkpoint)

    for kept, original in zip(_stats(target.model.running_mean_std),
                              _stats(source.model.running_mean_std)):
        assert torch.equal(kept, original)
    assert not _is_initial(source.model.value_mean_std)   # the source really had stats
    assert _is_initial(target.model.value_mean_std)


def test_warm_start_normalizer_flags_reverse_cleanly(tmp_path):
    source = _trained_source()
    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent(reset_obs_normalizer=True, reset_value_normalizer=False)
    A2CBase.set_warmstart_weights(target, checkpoint)

    assert _is_initial(target.model.running_mean_std)
    for kept, original in zip(_stats(target.model.value_mean_std),
                              _stats(source.model.value_mean_std)):
        assert torch.equal(kept, original)


def test_warm_start_resets_dict_observation_normalizers(tmp_path):
    """RunningMeanStdObs is a ModuleDict of RunningMeanStd - reset must recurse."""
    source = _trained_source(dict_obs=True)
    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent(dict_obs=True, reset_obs_normalizer=True)
    A2CBase.set_warmstart_weights(target, checkpoint)

    per_key = target.model.running_mean_std.running_mean_std
    assert set(per_key) == {'a', 'b'}
    for sub in per_key.values():
        assert _is_initial(sub)


def test_warm_start_keeps_dict_observation_normalizers_by_default(tmp_path):
    source = _trained_source(dict_obs=True)
    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent(dict_obs=True)
    A2CBase.set_warmstart_weights(target, checkpoint)

    for key, sub in target.model.running_mean_std.running_mean_std.items():
        original = source.model.running_mean_std.running_mean_std[key]
        for kept, was in zip(_stats(sub), _stats(original)):
            assert torch.equal(kept, was)


class _CentralValueNet(nn.Module):
    """Enough of CentralValueTrain for the ordering test below.

    It owns its own normalizers, and its state dict is loaded separately from
    the agent's model - which is exactly what makes the order matter.
    """

    def __init__(self):
        super().__init__()
        self.model = _Model()

    def get_stats_weights(self, model_stats=False):
        # The real CentralValueTrain contributes a 'central_val_stats' entry;
        # nothing under test reads it back (its set_stats_weights is a no-op).
        return {}


def test_warm_start_resets_the_central_value_nets_normalizers_too(tmp_path):
    """The central value net is loaded from its own key in the checkpoint.

    Reset the normalizers before that load and `assymetric_vf_nets` puts the old
    statistics straight back, silently - the flags would appear to work for the
    plain case and do nothing here.
    """
    source = _Agent()
    source.has_central_value = True
    source.central_value_net = _CentralValueNet()
    _feed(source.central_value_net.model.running_mean_std, 11.0)
    _feed(source.central_value_net.model.value_mean_std, 13.0)
    _feed(source.model.running_mean_std, 3.0)
    _feed(source.model.value_mean_std, 7.0)
    source.epoch_num = 5

    checkpoint = _checkpoint(source, tmp_path)
    assert 'assymetric_vf_nets' in checkpoint

    target = _Agent(reset_obs_normalizer=True, reset_value_normalizer=True)
    target.has_central_value = True
    target.central_value_net = _CentralValueNet()
    # The alias a2c_continuous sets when there is a central value net.
    target.value_mean_std = target.central_value_net.model.value_mean_std
    A2CBase.set_warmstart_weights(target, checkpoint)

    assert _is_initial(target.model.running_mean_std)
    assert _is_initial(target.central_value_net.model.running_mean_std)
    assert _is_initial(target.central_value_net.model.value_mean_std)


def test_central_value_net_weights_still_transfer(tmp_path):
    """...while the weights it was loaded for do come across."""
    source = _Agent()
    source.has_central_value = True
    source.central_value_net = _CentralValueNet()
    with torch.no_grad():
        source.central_value_net.model.a2c_network.weight.fill_(0.75)
    source.epoch_num = 5

    checkpoint = _checkpoint(source, tmp_path)

    target = _Agent()
    target.has_central_value = True
    target.central_value_net = _CentralValueNet()
    target.value_mean_std = target.central_value_net.model.value_mean_std
    A2CBase.set_warmstart_weights(target, checkpoint)

    assert torch.equal(target.central_value_net.model.a2c_network.weight,
                       source.central_value_net.model.a2c_network.weight)


def test_running_mean_std_reset_matches_a_fresh_module():
    rms = RunningMeanStd((OBS_SIZE,))
    fresh = RunningMeanStd((OBS_SIZE,))

    _feed(rms, 5.0)
    assert not _is_initial(rms)

    rms.reset()
    for after, expected in zip(_stats(rms), _stats(fresh)):
        assert torch.equal(after, expected)
        assert after.dtype == expected.dtype


# --------------------------------------------------------------------------- #
# 10. Config validation                                                         #
# --------------------------------------------------------------------------- #

def _parse_warm_start_config(block):
    """The validation A2CBase.__init__ performs, against a stub `self`.

    Reproduced rather than driven through __init__ because constructing an agent
    needs an environment; the assertions below are what the config block is for.
    """
    unknown = sorted(set(block) - {'enabled'} - set(WARM_START_CONFIG_KEYS))
    if unknown:
        raise ValueError(
            'Unknown key(s) in config.warm_start: {}. Supported keys: {}.'.format(
                ', '.join(unknown), ', '.join(('enabled',) + WARM_START_CONFIG_KEYS)
            )
        )
    count = int(block.get('critic_warmup_epoch_count', 0))
    if count < 0:
        raise ValueError(
            'config.warm_start.critic_warmup_epoch_count must be >= 0, got {}.'.format(count)
        )
    if count > 0 and not bool(block.get('enabled', False)):
        raise ValueError(
            'config.warm_start.critic_warmup_epoch_count is {} but '
            'config.warm_start.enabled is false - there is no transfer to warm the '
            'critic up after. Enable warm start or set the count to 0.'.format(count)
        )
    return block


def test_unknown_warm_start_key_is_rejected_by_name():
    try:
        _parse_warm_start_config({'enabled': True, 'reset_optimiser': True})
    except ValueError as e:
        assert 'reset_optimiser' in str(e)
    else:
        assert False, 'expected ValueError naming the misspelled key'


def test_every_documented_key_is_accepted():
    _parse_warm_start_config({key: 0 if key.endswith('count') else True
                              for key in ('enabled',) + WARM_START_CONFIG_KEYS})


def test_critic_warmup_epoch_count_is_accepted():
    _parse_warm_start_config({'enabled': True, 'critic_warmup_epoch_count': 3})


def test_critic_warmup_epoch_count_zero_is_accepted():
    _parse_warm_start_config({'enabled': True, 'critic_warmup_epoch_count': 0})


def test_negative_critic_warmup_epoch_count_is_rejected():
    try:
        _parse_warm_start_config({'enabled': True, 'critic_warmup_epoch_count': -1})
    except ValueError as e:
        assert 'critic_warmup_epoch_count' in str(e)
    else:
        assert False, 'expected ValueError'


def test_critic_warmup_without_warm_start_is_rejected():
    # The window is anchored to the epoch a transfer restores, so asking for one
    # without a transfer is a config mistake rather than a silent no-op.
    try:
        _parse_warm_start_config({'enabled': False, 'critic_warmup_epoch_count': 3})
    except ValueError as e:
        assert 'enabled' in str(e)
    else:
        assert False, 'expected ValueError'


# --------------------------------------------------------------------------- #
# 11. torch_runner dispatch                                                     #
# --------------------------------------------------------------------------- #

class _RecordingAgent:
    def __init__(self, warm_start_enabled=False, supports_warm_start=True):
        self.warm_start_enabled = warm_start_enabled
        self.calls = []
        if supports_warm_start:
            self.load_warmstart = lambda fn: self.calls.append(('load_warmstart', fn))

    def restore(self, fn):
        self.calls.append(('restore', fn))


class _RecordingPlayer:
    """A player: no warm_start_enabled attribute at all."""

    def __init__(self):
        self.calls = []

    def restore(self, fn):
        self.calls.append(('restore', fn))


def test_dispatch_resumes_when_warm_start_is_off():
    agent = _RecordingAgent(warm_start_enabled=False)
    _restore(agent, {'checkpoint': 'ckpt.pth'})
    assert agent.calls == [('restore', 'ckpt.pth')]


def test_dispatch_transfers_when_warm_start_is_on():
    agent = _RecordingAgent(warm_start_enabled=True)
    _restore(agent, {'checkpoint': 'ckpt.pth'})
    assert agent.calls == [('load_warmstart', 'ckpt.pth')]


def test_dispatch_leaves_players_alone():
    player = _RecordingPlayer()
    _restore(player, {'checkpoint': 'ckpt.pth'})
    assert player.calls == [('restore', 'ckpt.pth')]


def test_dispatch_without_a_checkpoint_does_nothing():
    agent = _RecordingAgent(warm_start_enabled=False)
    _restore(agent, {})
    _restore(agent, {'checkpoint': None})
    _restore(agent, {'checkpoint': ''})
    assert agent.calls == []


def test_warm_start_without_a_checkpoint_is_an_error():
    agent = _RecordingAgent(warm_start_enabled=True)
    for args in ({}, {'checkpoint': None}, {'checkpoint': ''}):
        try:
            _restore(agent, args)
        except ValueError as e:
            assert 'no checkpoint' in str(e)
        else:
            assert False, 'expected ValueError for warm start without a checkpoint'


def test_warm_start_on_an_agent_that_cannot_do_it():
    agent = _RecordingAgent(warm_start_enabled=True, supports_warm_start=False)
    try:
        _restore(agent, {'checkpoint': 'ckpt.pth'})
    except NotImplementedError as e:
        assert 'warm start is not supported' in str(e)
    else:
        assert False, 'expected NotImplementedError'


# --------------------------------------------------------------------------- #
# 12. Critic warmup                                                             #
# --------------------------------------------------------------------------- #

def _a2c_network(separate, fixed_sigma=True):
    """A real A2CBuilder network, so the freeze is tested against real modules.

    The partition is the whole point of the feature, and it turns on attributes
    (`separate`, `fixed_sigma`, the empty critic_mlp when not separate) that only
    the real builder produces.
    """
    from rl_games.algos_torch.network_builder import A2CBuilder

    builder = A2CBuilder()
    builder.load({
        'separate': separate,
        'mlp': {
            'units': [8, 8],
            'activation': 'elu',
            'initializer': {'name': 'default'},
        },
        'space': {
            'continuous': {
                'mu_activation': 'None',
                'sigma_activation': 'None',
                'mu_init': {'name': 'default'},
                'sigma_init': {'name': 'const_initializer', 'val': 0},
                'fixed_sigma': fixed_sigma,
            }
        },
    })
    return builder.build('a2c', actions_num=2, input_shape=(OBS_SIZE,), num_seqs=1, value_size=1)


class _NetworkAgent(_Agent):
    """An _Agent whose model carries a real A2CBuilder network."""

    def __init__(self, separate, fixed_sigma=True, **kwargs):
        super().__init__(**kwargs)
        self.model.a2c_network = _a2c_network(separate, fixed_sigma=fixed_sigma)


def _trainable(agent):
    return {name for name, param in agent.model.a2c_network.named_parameters()
            if param.requires_grad}


def test_warmup_anchor_is_relative_to_the_restored_epoch():
    # The transfer inherits the checkpoint's epoch counter, so a window measured
    # from 0 would already be over before the first epoch of any later iteration.
    source = _trained_source()
    source.epoch_num = 700
    saved = source.save_state()

    agent = _Agent(critic_warmup_epoch_count=10)
    agent.load_warm(saved)

    assert agent.epoch_num == 700
    assert agent.critic_warmup_until_epoch == 710


def test_no_anchor_without_a_warmup_count():
    source = _trained_source()
    source.epoch_num = 700
    agent = _Agent(critic_warmup_epoch_count=0)
    agent.load_warm(source.save_state())

    assert agent.critic_warmup_until_epoch is None


def test_warmup_state_machine_spans_exactly_the_window():
    agent = _NetworkAgent(separate=False)
    agent.critic_warmup_until_epoch = 705
    agent.epoch_num = 700

    seen = []
    for epoch in range(701, 709):
        agent.epoch_num = epoch
        agent._update_critic_warmup_state()
        seen.append(agent.critic_warmup_active)

    # Epochs 701..705 are critic-only; 706 onwards are normal.
    assert seen == [True] * 5 + [False] * 3


def test_warmup_never_activates_without_an_anchor():
    agent = _NetworkAgent(separate=False)
    for epoch in range(1, 5):
        agent.epoch_num = epoch
        agent._update_critic_warmup_state()
        assert not agent.critic_warmup_active


def test_shared_trunk_freezes_the_trunk_and_leaves_only_the_value_head():
    # The case the ARD tasks actually run: separate=False, so actor_mlp *is* the
    # trunk both heads read, and a critic-only epoch may only move `value`.
    agent = _NetworkAgent(separate=False)
    agent.critic_warmup_until_epoch = 5
    agent.epoch_num = 1
    agent._update_critic_warmup_state()

    assert agent.critic_warmup_active
    assert _trainable(agent) == {'value.weight', 'value.bias'}


def test_separate_trunks_keep_the_critic_trunk_trainable():
    agent = _NetworkAgent(separate=True)
    agent.critic_warmup_until_epoch = 5
    agent.epoch_num = 1
    agent._update_critic_warmup_state()

    trainable = _trainable(agent)
    assert {'value.weight', 'value.bias'} <= trainable
    assert any(name.startswith('critic_mlp.') for name in trainable)
    assert not any(name.startswith('actor_mlp.') for name in trainable)
    assert not any(name.startswith('mu.') for name in trainable)
    assert not any(name.startswith('sigma') for name in trainable)


def test_fixed_sigma_parameter_is_frozen():
    # fixed_sigma makes sigma a bare nn.Parameter rather than a Linear, which a
    # module-name-based freeze would walk straight past.
    for separate in (False, True):
        agent = _NetworkAgent(separate=separate, fixed_sigma=True)
        assert agent.model.a2c_network.sigma.requires_grad

        agent.critic_warmup_until_epoch = 5
        agent.epoch_num = 1
        agent._update_critic_warmup_state()

        assert not agent.model.a2c_network.sigma.requires_grad


def test_unfreeze_restores_the_original_flags():
    agent = _NetworkAgent(separate=False)
    before = {name: param.requires_grad
              for name, param in agent.model.named_parameters()}

    agent.critic_warmup_until_epoch = 5
    agent.epoch_num = 1
    agent._update_critic_warmup_state()
    agent.epoch_num = 6
    agent._update_critic_warmup_state()

    assert not agent.critic_warmup_active
    after = {name: param.requires_grad
             for name, param in agent.model.named_parameters()}
    assert after == before


def test_unfreeze_does_not_thaw_what_was_already_frozen():
    agent = _NetworkAgent(separate=False)
    agent.model.a2c_network.mu.weight.requires_grad = False

    agent.critic_warmup_until_epoch = 5
    agent.epoch_num = 1
    agent._update_critic_warmup_state()
    agent.epoch_num = 6
    agent._update_critic_warmup_state()

    assert not agent.model.a2c_network.mu.weight.requires_grad


def test_schedule_is_frozen_during_warmup():
    # KL is 0 by construction while the policy is held still, so an adaptive
    # scheduler would ramp the lr through the whole window.
    agent = _NetworkAgent(separate=False)
    agent.critic_warmup_active = True
    agent._maybe_update_schedule(0.0)

    assert agent.scheduler.calls == 0
    assert agent.last_lr == CONFIG_LR


def test_schedule_advances_once_warmup_is_over():
    agent = _NetworkAgent(separate=False)
    agent.critic_warmup_active = False
    agent._maybe_update_schedule(0.01)

    assert agent.scheduler.calls == 1
    assert agent.last_lr == CONFIG_LR * 2.0


def test_unsupported_network_is_rejected_by_name():
    # _Agent's plain nn.Linear stands in for any network without the
    # actor_mlp/value split the freeze needs.
    agent = _Agent()
    try:
        agent._critic_warmup_frozen_parameters()
    except NotImplementedError as e:
        assert 'critic_warmup_epoch_count' in str(e)
    else:
        assert False, 'expected NotImplementedError naming the unsupported network'
