import brax.jumpy as jp
import brax
from diffmimic.utils.io import deserialize_qp, serialize_qp
import numpy as np

def add_box(qp, box_qp):
    return jp.tree_map(lambda x, y: np.concatenate([x,y], axis=-2), qp, box_qp)

def remove_box(qp):
    return jp.tree_map(lambda x: x[..., :19, :], qp), jp.tree_map(lambda x: x[..., 19:, :], qp)

def add_scene_to_traj(traj, waypoint, scene='hit', bar_height=1.25):
    if scene=='hit':
        x_ed, y_ed = waypoint
        target_qp = brax.QP.zero(shape=(traj.shape[0], traj.shape[1], 1,))
        target_qp.pos[..., 0]=x_ed
        target_qp.pos[..., 1]=y_ed
        target_qp.pos[..., 2]=0.8
        combined_traj = serialize_qp(add_box(deserialize_qp(traj), target_qp))

        box_qp = brax.QP.zero(shape=(traj.shape[0], traj.shape[1], 1,))
        box_qp.pos[..., 0]=100
        box_qp.pos[..., 1]=100
        box_qp.pos[..., 2]=0.1
        combined_traj = serialize_qp(add_box(deserialize_qp(combined_traj), box_qp))

        return combined_traj
    elif scene=='vis':
        target_qp = brax.QP.zero(shape=(traj.shape[0], traj.shape[1], 1,))
        target_qp.pos[..., 0]=-100
        target_qp.pos[..., 1]=-100
        target_qp.pos[..., 2]=0.8
        combined_traj = serialize_qp(add_box(deserialize_qp(traj), target_qp))

        box_qp = brax.QP.zero(shape=(traj.shape[0], traj.shape[1], 1,))
        box_qp.pos[..., 0]=100
        box_qp.pos[..., 1]=100
        box_qp.pos[..., 2]=0.1
        combined_traj = serialize_qp(add_box(deserialize_qp(combined_traj), box_qp))

        return combined_traj


def remove_scene_from_traj(traj):
    qp, box_qp = remove_box(deserialize_qp(traj))
    traj = serialize_qp(qp)
    box_traj = serialize_qp(box_qp)
    return traj, box_traj
