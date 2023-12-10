import brax
from brax import jumpy as jp
from brax.envs import env
from diffmimic.mimic_envs.system_configs import get_system_cfg
from diffmimic.utils.io import deserialize_qp, serialize_qp
from diffmimic.mimic_envs.losses import *
from diffmimic.utils.rotation6d import quaternion_to_rotation_6d
from diffmimic.utils.quaternion import *
import jax
import jax.numpy as jnp

from google.protobuf import text_format
from brax.physics import config_pb2


class HumanoidMimic(env.Env):
    """Trains a humanoid to mimic reference motion."""

    def __init__(self, system_config, reward_scaling=1., vel_weight=0., rot_weight=1., ang_weight=1., local='none',
                 perturb=False):
        # super().__init__(config=get_system_cfg(system_config))
        self.reward_scaling = reward_scaling
        self.vel_weight = vel_weight
        self.rot_weight = rot_weight
        self.ang_weight = ang_weight
        self.local = local
        self.perturb = perturb

        config = text_format.Parse(get_system_cfg(system_config), brax.Config())


        target = config.bodies.add()
        target.name = 'target'
        target.colliders.add(box=config_pb2.Collider.Box(halfsize=config_pb2.Vector3(x=0.1, y=0.1, z=0.8)))
        target.mass = 1.0
        target.inertia.MergeFrom(config_pb2.Vector3(x=1, y=1, z=1))
        for obj in config.bodies:
            ci = config.collide_include.add()
            ci.first = obj.name
            ci.second = target.name

        box = config.bodies.add()
        box.name = 'target_2'
        box.colliders.add(box=config_pb2.Collider.Box(halfsize=config_pb2.Vector3(x=0.1, y=0.1, z=0.1)))
        box.mass = 2.0
        box.inertia.MergeFrom(config_pb2.Vector3(x=1, y=1, z=1))
        for obj in config.bodies:
            ci = config.collide_include.add()
            ci.first = obj.name
            ci.second = box.name

        self.sys = brax.System(config)

    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        qp = self.sys.default_qp()
        ref_traj = serialize_qp(qp)[None, ...].repeat(2, axis=0)
        mask = jp.zeros(2)
        return self.reset_ref(rng, ref_traj, mask)

    def reset_ref(self, rng: jp.ndarray, ref_traj: jp.ndarray, mask: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""
        reward, done, zero = jp.zeros(3)

        info = {'reference_qp': deserialize_qp(ref_traj), 'mask': mask}
        qp = self._get_ref_state(zero, info['reference_qp'])
        target_qp = self._get_ref_state(step_idx=1, reference_qp=info['reference_qp'])
        metrics = {'step_index': zero, 'fall': zero, 'kl_div': zero,
                   # 'pose_error': zero,
                   'loss_pos': zero,
                   'loss_rot': zero,
                   'loss_vel': zero,
                   'loss_ang': zero
                   }
        obs = self._get_obs(qp, target_qp)
        info['obs'] = obs
        info['key'] = rng
        state = env.State(qp, obs, reward, done, metrics, info)
        return state


    def throw_box(self, box_qp, i, root_pos, key):
        target_pos = jnp.zeros(root_pos.shape)
        target_pos = target_pos.at[..., :2].set(root_pos[..., :2])
        tmp_key, key = jax.random.split(key)
        target_pos = target_pos.at[..., 2].set(root_pos[..., 2] + jax.random.uniform(tmp_key, minval=0., maxval=1., shape=target_pos[..., 2].shape))

        displacement = jnp.zeros(root_pos.shape) 
        tmp_key, key = jax.random.split(key)
        theta =  jax.random.uniform(tmp_key, minval=-jnp.pi, maxval=jnp.pi, shape=target_pos[..., 2].shape)
        radius = 2
        displacement = displacement.at[..., 0].set(radius * jnp.sin(theta))
        displacement = displacement.at[..., 1].set(radius * jnp.cos(theta))
        displacement = displacement.at[..., 2].set(0.5)

        direction = displacement / jnp.linalg.norm(displacement, ord=2, axis=-1, keepdims=True)

        box_pos = target_pos + displacement
        box_vel = -1 * direction * 5.0
        pos = box_qp.pos
        vel = box_qp.vel
        pos = pos.at[..., i, :].set(box_pos)
        vel = vel.at[..., i, :].set(box_vel)
        box_qp = box_qp.replace(pos=pos, vel=vel)
        return box_qp
    
    def remove_box(self, qp):
        return jp.tree_map(lambda x: x[..., :-1, :], qp), jp.tree_map(lambda x: x[..., -1:, :], qp)

    def add_box(self, qp, box_qp):
        return jp.tree_map(lambda x, y: jnp.concatenate([x,y], axis=-2), qp, box_qp)

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""
        step_index = state.metrics['step_index'] + 1
        qp, info = self.sys.step(state.qp, action)
        target_qp = self._get_ref_state(step_idx=step_index + 1, reference_qp=state.info['reference_qp'])
        obs = self._get_obs(qp, target_qp)
        ref_qp = self._get_ref_state(step_idx=step_index, reference_qp=state.info['reference_qp'])
        state.info['obs'] = self._get_obs(ref_qp, target_qp)

        if self.perturb:
            throw_key, key = jax.random.split(state.info['key'])
            state.info.update(key=key)
            body_qp, box_qp = self.remove_box(qp)
            box_qp = self.throw_box(box_qp, 0, body_qp.pos[..., 0, :], throw_key)
            qp_after_throw=self.add_box(body_qp, box_qp)
            throw = jp.where(step_index % 30 == 0, jp.float32(1), jp.float32(0))
            qp = jp.tree_map(lambda x: x * throw, qp_after_throw) + jp.tree_map(lambda x: x * (1 - throw), qp)

        qp_normalized = convert_to_local(qp)
        ref_qp_normalized = convert_to_local(ref_qp)
        loss_pos = loss_l2_pos(qp_normalized, ref_qp_normalized) ** 0.5
        loss_rot = loss_l2_rot(qp_normalized, ref_qp_normalized) ** 0.5
        loss_vel = loss_l2_vel(qp_normalized, ref_qp_normalized) ** 0.5
        loss_ang = loss_l2_ang(qp_normalized, ref_qp_normalized) ** 0.5
        state.metrics.update(
            loss_pos=jnp.copy(jax.lax.stop_gradient(loss_pos)),
            loss_rot=jnp.copy(jax.lax.stop_gradient(loss_rot)),
            loss_vel=jnp.copy(jax.lax.stop_gradient(loss_vel)),
            loss_ang=jnp.copy(jax.lax.stop_gradient(loss_ang))
        )

        if self.local == 'st':
            reward = -1 * (loss_pos +
                           self.rot_weight * loss_rot +
                           self.vel_weight * loss_vel +
                           self.ang_weight * loss_ang
                           ) * self.reward_scaling
            # pose_error = jax.lax.stop_gradient(loss_pos)
        elif self.local == 'dm':
            reward = -1 * (loss_pos +
                           self.rot_weight * loss_rot +
                           self.vel_weight * loss_vel +
                           self.ang_weight * loss_ang
                           ) * self.reward_scaling * 0.9

            reward += -1 * (loss_l2_pos(qp, ref_qp) ** 0.5 +
                           self.rot_weight * loss_l2_rot(qp, ref_qp) ** 0.5 +
                           self.vel_weight * loss_l2_vel(qp, ref_qp) ** 0.5 +
                           self.vel_weight * loss_l2_ang(qp, ref_qp) ** 0.5
                           ) * self.reward_scaling * 0.1
            # pose_error = jax.lax.stop_gradient(loss_pos)
        else:
            reward = -1 * (loss_l2_pos(qp, ref_qp) ** 0.5 +
                           self.rot_weight * loss_l2_rot(qp, ref_qp) ** 0.5 +
                           self.vel_weight * loss_l2_vel(qp, ref_qp) ** 0.5 +
                           self.ang_weight * loss_l2_ang(qp, ref_qp) ** 0.5
                           ) * self.reward_scaling
            # pose_error = loss_l1_relpos(qp, ref_qp)

        fall = jp.where(qp.pos[0, 2] < 0.2, jp.float32(1), jp.float32(0))
        fall = jp.where(qp.pos[0, 2] > 1.7, jp.float32(1), fall)
        state.metrics.update(
            step_index=step_index,
            # pose_error=jnp.copy(pose_error),
            fall=fall
        )
        state = state.replace(qp=qp, obs=obs, reward=reward)
        state = self._mask_state(state)
        return state

    def _get_obs(self, qp: brax.QP, target_qp: brax.QP) -> jp.ndarray:
        """Observe humanoid body position, velocities, and angles."""
        if self.local == 'st':
            qp_normalized = convert_to_local(qp)
            target_qp_normalized = convert_to_local(target_qp)
        elif self.local == 'dm':
            # qp_normalized = convert_to_local(qp, remove_height=True)
            # target_qp_normalized = convert_to_local(target_qp, remove_height=True)
            qp_normalized, target_qp_normalized = convert_pair_to_local(qp, target_qp)
        else:
            qp_normalized, target_qp_normalized = convert_pair_to_local(qp, target_qp)
        pos, rot, vel, ang = qp_normalized.pos[:18], quaternion_to_rotation_6d(
            qp_normalized.rot[:18]), qp_normalized.vel[:18], qp_normalized.ang[:18]
        target_pos, target_rot, target_vel, target_ang = target_qp_normalized.pos[:18], quaternion_to_rotation_6d(
            target_qp_normalized.rot[:18]), target_qp_normalized.vel[:18], target_qp_normalized.ang[:18]

        obs = jp.concatenate([pos.reshape(-1), rot.reshape(-1), vel.reshape(-1), ang.reshape(-1),
                              target_pos.reshape(-1), target_rot.reshape(-1), target_vel.reshape(-1),
                              target_ang.reshape(-1)
                              ],
                             axis=-1)
        return obs

    def _get_ref_state(self, step_idx, reference_qp) -> brax.QP:
        reference_len = reference_qp.pos.shape[0]
        mask = jp.where(step_idx == jp.arange(0, reference_len), jp.float32(1), jp.float32(0))
        ref_state = jp.tree_map(lambda x: (mask @ x.transpose(1, 0, 2)), reference_qp)
        return ref_state

    # def _mask_state(self, state) -> brax.QP:
    #     _, zero = jp.zeros(2)
    #     mask = jp.where(state.metrics['step_index'] == jp.arange(0, state.info['mask'].shape[0]), jp.float32(1), jp.float32(0))
    #     state_mask = mask@state.info['mask']
    #     reward = jp.where(state_mask == 0, zero, state.reward)
    #     pose_error = jp.where(state_mask == 0, zero, state.metrics['pose_error'])
    #     state = state.replace(reward=reward)
    #     state.metrics.update(
    #         pose_error=pose_error,
    #     )
    #     return state

    def _mask_state(self, state) -> brax.QP:
        _, zero = jp.zeros(2)
        mask = jp.where((state.metrics['step_index'] + 1) == jp.arange(0, state.info['mask'].shape[0]), jp.float32(1),
                        jp.float32(0))
        valid = mask @ state.info['mask']
        init_qp = self._get_ref_state(0, state.info['reference_qp'])
        target_qp = self._get_ref_state(1, state.info['reference_qp'])
        init_obs = self._get_obs(init_qp, target_qp)
        qp = jp.tree_map(lambda x: x * valid, state.qp) + jp.tree_map(lambda x: x * (1 - valid), init_qp)
        obs = state.obs * valid + init_obs * (1 - valid)
        # obs = self._get_obs(init_qp, target_qp)
        state.metrics.update(
            step_index=state.metrics['step_index'] * valid,
        )
        state = state.replace(qp=qp, obs=obs)
        return state
