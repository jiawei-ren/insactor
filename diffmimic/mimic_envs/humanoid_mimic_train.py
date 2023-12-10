from brax import jumpy as jp
from brax.envs import env
from .humanoid_mimic import HumanoidMimic
from .losses import *
import jax


class HumanoidMimicTrain(HumanoidMimic):
    """Trains a humanoid to mimic reference motion."""

    def __init__(self, system_config,
                 early_termination, demo_replay_mode, err_threshold, replay_rate, **kwargs):
        super().__init__(system_config, **kwargs)
        self.early_termination = early_termination
        self.demo_replay_mode = demo_replay_mode
        self.err_threshold = err_threshold
        self.replay_rate = replay_rate

    def reset_ref(self, rng: jp.ndarray, ref_traj: jp.ndarray, mask: jp.ndarray) -> env.State:
        state = super(HumanoidMimicTrain, self).reset_ref(rng, ref_traj, mask)
        state.metrics.update(replay=jp.zeros(1)[0])
        state.info.update(replay_key=rng)
        return state

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        state = super(HumanoidMimicTrain, self).step(state, action)
        if self.early_termination:
            state = state.replace(done=state.metrics['fall'])
        state = self._demo_replay(state)
        # state = self._mask_state(state)
        return state

    def _demo_replay(self, state) -> env.State:
        qp = state.qp
        ref_qp = self._get_ref_state(state.metrics['step_index'], state.info['reference_qp'])

        if self.demo_replay_mode == 'threshold':
            if self.local == 'st':
                qp_normalized = convert_to_local(qp)
                ref_qp_normalized = convert_to_local(ref_qp)
                error = loss_l1_pos(qp_normalized, ref_qp_normalized)
            else:
                error = loss_l1_pos(qp, ref_qp)
            replay = jp.where(error > self.err_threshold, jp.float32(1), jp.float32(0))
        elif self.demo_replay_mode == 'random':
            replay_key, key = jax.random.split(state.info['replay_key'])
            state.info.update(replay_key=replay_key)
            replay = jp.where(jax.random.bernoulli(key, p=self.replay_rate), jp.float32(1), jp.float32(0))
        else:
            raise NotImplementedError
        qp = jp.tree_map(lambda x: x*(1 - replay), qp) + jp.tree_map(lambda x: x*replay, ref_qp)
        target_qp = self._get_ref_state(step_idx=state.metrics['step_index'] + 1, reference_qp=state.info['reference_qp'])
        obs = self._get_obs(qp, target_qp)
        state.metrics.update(replay=replay)
        return state.replace(qp=qp, obs=obs)
