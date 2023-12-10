from brax import jumpy as jp
from brax.envs import env as brax_env
import jax
from brax.envs import wrappers


class EvalWrapper(wrappers.EvalWrapper):
  """Brax env with eval metrics."""

  def reset_ref(self, rng: jp.ndarray, ref_traj: jp.ndarray, mask: jp.ndarray, text_embedding: jp.ndarray) -> brax_env.State:

    reset_state = self.env.reset_ref(rng, ref_traj, mask, text_embedding)
    reset_state.metrics['reward'] = reset_state.reward
    eval_metrics = wrappers.EvalMetrics(
        episode_metrics=jax.tree_util.tree_map(jp.zeros_like,
                                               reset_state.metrics),
        active_episodes=jp.ones_like(reset_state.reward),
        episode_steps=jp.zeros_like(reset_state.reward))
    reset_state.info['eval_metrics'] = eval_metrics
    return reset_state


class AutoResetWrapper(wrappers.AutoResetWrapper):
  """Automatically resets Brax envs that are done."""

  def reset_ref(self, rng: jp.ndarray, ref_traj: jp.ndarray, mask: jp.ndarray, text_embedding: jp.ndarray) -> brax_env.State:
    state = self.env.reset_ref(rng, ref_traj, mask, text_embedding)
    state.info['first_qp'] = state.qp
    state.info['first_obs'] = state.obs
    return state


class EpisodeWrapper(wrappers.EpisodeWrapper):
  """Maintains episode step count and sets done at episode end."""

  def reset_ref(self, rng: jp.ndarray, ref_traj: jp.ndarray, mask: jp.ndarray, text_embedding: jp.ndarray) -> brax_env.State:
    state = self.env.reset_ref(rng, ref_traj, mask, text_embedding)
    state.info['steps'] = jp.zeros(())
    state.info['truncation'] = jp.zeros(())
    return state


class VmapWrapper(wrappers.VmapWrapper):
  """Vectorizes Brax env."""

  def reset_ref(self, rng: jp.ndarray, ref_traj: jp.ndarray, mask: jp.ndarray, text_embedding:jp.ndarray) -> brax_env.State:
    return jp.vmap(self.env.reset_ref)(rng, ref_traj, mask, text_embedding)