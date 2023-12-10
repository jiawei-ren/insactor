import numpy as np
from brax import envs
import brax

import jax
import jax.numpy as jnp

from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.apg import networks as apg_networks
import brax.jumpy as jp

from diffmimic.mimic_envs import register_mimic_env
from diffmimic.utils.io import serialize_qp, deserialize_qp
from diffmimic.brax_lib import wrappers
import diffmimic.brax_lib.encoder as encoder_networks
from tqdm import tqdm
from diffmimic.utils.quaternion import *
from diffmimic.brax_lib import acting
import functools

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from simulate.humanoid_mimic_hit import HumanoidMimic as HumanoidMimicHit
envs.register_environment('humanoid_mimic_hit', HumanoidMimicHit)

  
env_global = envs.get_environment(
    env_name="humanoid_mimic_hit",
    system_config='smpl',
)

env_global_perturb = envs.get_environment(
    env_name="humanoid_mimic_hit",
    system_config='smpl',
    perturb=True
)



# param_path = 'pretrained_models/skill_human_0.0.pkl'
param_path = 'pretrained_models/skill_human_0.001.pkl'
params_global = model.load_params(param_path)
latent_dim = 64


skip_encoder = '0.0.' in param_path
if skip_encoder:
    latent_dim = env_global.observation_size // 2

apg_network = apg_networks.make_apg_networks(
    env_global.observation_size // 2 + latent_dim,
    env_global.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    hidden_layer_sizes=(1024,) * 5,
)
make_policy = apg_networks.make_inference_fn(apg_network)


encoder_network = encoder_networks.make_encoder_networks(
    env_global.observation_size,
    latent_dim,
    hidden_layer_sizes=(1024,) * 5,
    preprocess_observations_fn=running_statistics.normalize,
)

make_encoder = encoder_networks.make_inference_fn(encoder_network)

def make_evaluator(env):

    ep_len_eval = 196

    eval_env = wrappers.EpisodeWrapper(env, ep_len_eval - 1, 1)
    eval_env = wrappers.VmapWrapper(eval_env)
    eval_env = wrappers.AutoResetWrapper(eval_env)
    return acting.Evaluator(
        eval_env,
        eval_policy_fn=functools.partial(make_policy, deterministic=False),
        eval_encoder_fn=functools.partial(make_encoder, deterministic=False),
        num_eval_envs=4,
        episode_length=ep_len_eval,
        action_repeat=1,
        key=jax.random.PRNGKey(999)
    )

evaluator_global = make_evaluator(env_global)
evaluator_global_perturb = make_evaluator(env_global_perturb)


def execute_actions(rollout_traj, perturb=False):
    evaluator = evaluator_global_perturb if perturb else evaluator_global
    metrics, (qp_list, latent_list) = evaluator.run_evaluation(
    params_global,
    ref_traj=rollout_traj,
    mask=np.ones(rollout_traj.shape[:-1]),
    training_metrics={})

    return serialize_qp(qp_list).transpose(1,0,2)[:, :rollout_traj.shape[1]]

if __name__ == '__main__':
    rollout_traj = np.zeros([4, 120, 273])
    a = execute_actions(rollout_traj)
    print(a.shape)