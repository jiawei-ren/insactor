# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
import jax
import numpy as np
import brax
import jax.numpy as jnp

from diffmimic.brax_lib import wrappers


def actor_step(
    env: envs.Env,
    env_state: envs.State,
    policy: Policy,
    encoder: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
):
  """Collect data."""
  key, key_encoder = jax.random.split(key)
  latent, kl_div = encoder(env_state.obs, key_encoder)
  obs_latent = jnp.concatenate([jnp.split(env_state.obs, 2, axis=-1)[0], latent], axis=-1)
  actions, policy_extras = policy(obs_latent, key)
  nstate = env.step(env_state, actions)
  nstate.metrics.update(kl_div=jax.lax.stop_gradient(kl_div))
  # nstate = nstate.replace(reward=nstate.reward - kl_div)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, (env_state.qp, latent)


def generate_unroll(
    env: envs.Env,
    env_state: envs.State,
    policy: Policy,
    encoder: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
):
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, (qp, latent) = actor_step(
        env, state, policy, encoder, current_key, extra_fields=extra_fields)
    return (nstate, next_key), (qp, latent)

  (final_state, _), (qp_list, latent_list) = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length)
  return final_state, (qp_list, latent_list)


# TODO: Consider moving this to its own file.
class Evaluator:
  """Class to run evaluations."""

  def __init__(self, eval_env: envs.Env,
               eval_policy_fn: Callable[[PolicyParams],
                                        Policy],
               eval_encoder_fn: Callable[[PolicyParams],
                                        Policy],
               num_eval_envs: int,
               episode_length: int, action_repeat: int, key: PRNGKey):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.

    # eval_env = envs.wrappers.EvalWrapper(eval_env)
    eval_env = wrappers.EvalWrapper(eval_env)

    def generate_eval_unroll(cvae_params: PolicyParams,
                             key: PRNGKey,
                             ref_traj: jnp.ndarray,
                             mask: jnp.ndarray) -> (envs.State, brax.QP):
      reset_keys = jax.random.split(key, num_eval_envs)
      # eval_first_state = eval_env.reset(reset_keys)
      eval_first_state = eval_env.reset_ref(reset_keys, ref_traj, mask)
      (normalizer_encoder, normalizer_policy), (encoder_params, policy_params) = cvae_params
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn((normalizer_policy, policy_params)),
          eval_encoder_fn((normalizer_encoder, encoder_params)),
          key,
          unroll_length=episode_length // action_repeat)

    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(self,
                     cvae_params: PolicyParams,
                     training_metrics: Metrics,
                     ref_traj: jnp.ndarray,
                     mask: jnp.ndarray,
                     aggregate_episodes: bool = True) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state, (qp_list, latent_list) = self._generate_eval_unroll(cvae_params, unroll_key, ref_traj, mask)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {
        f'eval/episode_{name}': np.mean(value) if aggregate_episodes else value
        for name, value in eval_metrics.episode_metrics.items()
    }
    metrics = {name: value/np.mean(eval_metrics.episode_steps) for name, value in metrics.items()}
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics
    }

    return metrics, (qp_list, latent_list)
