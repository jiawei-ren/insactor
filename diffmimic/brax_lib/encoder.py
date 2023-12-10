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

"""encoder networks."""

from typing import Sequence,Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

import jax.numpy as jnp
import jax


@flax.struct.dataclass
class EncoderNetworks:
  policy_network: networks.FeedForwardNetwork
  prior_network: networks.FeedForwardNetwork
  parametric_latent_distribution: distribution.ParametricDistribution
  conditional: bool
  random: bool


def make_inference_fn(encoder_networks: EncoderNetworks):
  """Creates params and inference function for the encoder agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      normalizer, (param_p, param_q) = params
      observations_prev, observations_curr = jnp.split(observations, 2, axis=-1)
      observation_p_padded = jnp.concatenate([observations_prev, jnp.zeros_like(observations_prev)], axis=-1)
      logits_p = encoder_networks.prior_network.apply(normalizer, param_p, observation_p_padded)
      logits_q = encoder_networks.policy_network.apply(normalizer, param_q, observations)

      if encoder_networks.conditional:

          loc_p, _ = jnp.split(logits_p, 2, axis=-1)
          loc_delta, scale_full = jnp.split(logits_q, 2, axis=-1)

          if not encoder_networks.random:
            loc = loc_p + loc_delta
          else:
            loc = loc_p + loc_delta * 0.
          scale = jnp.ones_like(scale_full) * scale_full[..., :1]

          logits = jnp.concatenate([loc, scale], axis=-1)

          sigma = jax.nn.softplus(scale) + 0.001
          kl_div = 0.5 * loc_delta ** 2 / sigma ** 2
          kl_div = kl_div.mean(-1)
          if deterministic:
            return loc, kl_div
          return encoder_networks.parametric_latent_distribution.sample_no_postprocessing(
              logits, key_sample), kl_div

      else:
          loc, scale = jnp.split(logits_q, 2, axis=-1)
          sigma = jax.nn.softplus(scale) + 0.001
          log_var = 2 * jnp.log(sigma)
          kl_div = 0.5 * (jnp.exp(log_var) + loc ** 2 - 1. - log_var)
          kl_div = kl_div.mean(-1)
          if deterministic:
              if observations_curr.shape[-1] == loc.shape[-1]:    # skip encoder
                return observations_curr, kl_div
              else:
                return loc, kl_div
          elif encoder_networks.random:
              return jax.random.normal(key_sample, shape=loc.shape), kl_div
          return encoder_networks.parametric_latent_distribution.sample_no_postprocessing(
              logits_q, key_sample), kl_div

    return policy

  return make_policy


def make_encoder_networks(
    observation_size: int,
    latent_size: int,
    conditional: bool = True,
    random: bool = False,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = linen.swish) -> EncoderNetworks:
  """Make encoder networks."""
  parametric_latent_distribution = distribution.NormalTanhDistribution(
      event_size=latent_size)
  policy_network = networks.make_policy_network(
      parametric_latent_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes, activation=activation)
  prior_network = networks.make_policy_network(
      parametric_latent_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes, activation=activation)
  return EncoderNetworks(
      policy_network=policy_network,
      prior_network=prior_network,
      parametric_latent_distribution=parametric_latent_distribution,
      conditional=conditional,
      random=random)
