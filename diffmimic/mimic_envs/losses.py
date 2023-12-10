from diffmimic.utils.rotation6d import quaternion_to_rotation_6d
from brax import QP
from diffmimic.utils.quaternion import *


def loss_l1_relpos(qp, ref_qp):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    relpos, ref_relpos = (pos - pos[0])[1:], (ref_pos - ref_pos[0])[1:]
    relpos_loss = (((relpos - ref_relpos) ** 2).sum(-1) ** 0.5).mean()
    relpos_loss = relpos_loss
    return relpos_loss


def loss_l2_pos(qp, ref_qp, reduce='mean'):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    pos_loss = ((pos - ref_pos) ** 2).sum(-1)
    if reduce == 'mean':
        pos_loss = pos_loss.mean()
    else:
        pos_loss = pos_loss.sum()
    return pos_loss


def loss_l1_pos(qp, ref_qp):
    pos, ref_pos = qp.pos[:-1], ref_qp.pos[:-1]
    pos_loss = (((pos - ref_pos) ** 2).sum(-1)**0.5).mean()
    return pos_loss


def loss_l2_rot(qp, ref_qp, reduce='mean'):
    rot, ref_rot = quaternion_to_rotation_6d(qp.rot[:-1]), quaternion_to_rotation_6d(ref_qp.rot[:-1])
    rot_loss = ((rot - ref_rot) ** 2).sum(-1)
    if reduce == 'mean':
        rot_loss = rot_loss.mean()
    else:
        rot_loss = rot_loss.sum()
    return rot_loss


def loss_l2_vel(qp, ref_qp, reduce='mean'):
    vel, ref_vel = qp.vel[:-1], ref_qp.vel[:-1]
    vel_loss = ((vel - ref_vel) ** 2).sum(-1)
    if reduce == 'mean':
        vel_loss = vel_loss.mean()
    else:
        vel_loss = vel_loss.sum()
    return vel_loss


def loss_l2_vel_foot(qp, ref_qp, reduce='mean'):
    vel, ref_vel = qp.vel[[14, 17], ...], ref_qp.vel[[14, 17], ...]
    vel_loss = ((vel - ref_vel) ** 2).sum(-1)
    if reduce == 'mean':
        vel_loss = vel_loss.mean()
    else:
        vel_loss = vel_loss.sum()
    return vel_loss


def loss_l1_vel(qp, ref_qp):
    vel, ref_vel = qp.vel[:-1], ref_qp.vel[:-1]
    vel_loss = (((vel - ref_vel) ** 2).sum(-1) ** 0.5).mean()
    return vel_loss



def loss_l2_ang(qp, ref_qp, reduce='mean'):
    ang, ref_ang = qp.ang[:-1], ref_qp.ang[:-1]
    ang_loss = ((ang - ref_ang) ** 2).sum(-1)
    if reduce == 'mean':
        ang_loss = ang_loss.mean()
    else:
        ang_loss = ang_loss.sum()
    return ang_loss


def loss_l1_ang(qp, ref_qp):
    ang, ref_ang = qp.ang[:-1], ref_qp.ang[:-1]
    ang_loss = (((ang - ref_ang) ** 2).sum(-1) ** 0.5).mean()
    return ang_loss



def loss_l1_rot(qp, ref_qp):
    rot, ref_rot = quaternion_to_rotation_6d(qp.rot[:-1]), quaternion_to_rotation_6d(ref_qp.rot[:-1])
    rot_loss = (((rot - ref_rot) ** 2).sum(-1)**0.5).mean()
    return rot_loss


def convert_to_local(qp, remove_height=False):
    pos, rot, vel, ang = qp.pos * 1., qp.rot * 1., qp.vel * 1., qp.ang * 1.

    # normalize
    if not remove_height:
        root_pos = pos[:1] * jp.array([1., 1., 0.])  # shape (1, 3)
    else:
        root_pos = pos[:1] * jp.array([1., 1., 1.])  # shape (1, 3)
    normalized_pos = pos - root_pos

    # normalize
    rot_xyzw_raw = rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    rot_xyzw = quat_normalize(rot_xyzw_raw)

    root_rot_xyzw = quat_normalize(rot_xyzw[:1] * jp.array([0., 0., 1., 1.]))  # [x, y, z, w] shape (1, 4)

    normalized_rot_xyzw = quat_mul_norm(quat_inverse(root_rot_xyzw), rot_xyzw)
    normalized_pos = quat_rotate(quat_inverse(root_rot_xyzw), normalized_pos)
    normalized_vel = quat_rotate(quat_inverse(root_rot_xyzw), vel)
    normalized_ang = quat_rotate(quat_inverse(root_rot_xyzw), ang)

    normalized_rot = normalized_rot_xyzw[:, [3, 0, 1, 2]]

    qp_normalized = QP(pos=normalized_pos, rot=normalized_rot, vel=normalized_vel, ang=normalized_ang)

    return qp_normalized


def convert_pair_to_local(qp, target_qp):
    pos, rot, vel, ang = qp.pos * 1., qp.rot * 1., qp.vel * 1., qp.ang * 1.
    target_pos, target_rot, target_vel, target_ang = target_qp.pos* 1., target_qp.rot* 1., target_qp.vel* 1., target_qp.ang* 1.

    # normalize
    root_pos = pos[:1] * jp.array([1., 1., 0.])  # shape (1, 3)
    normalized_pos = pos - root_pos
    normalized_target_pos = target_pos - root_pos

    # normalize
    rot_xyzw_raw = rot[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    target_rot_xyzw_raw = target_rot[:, [1, 2, 3, 0]]
    rot_xyzw = quat_normalize(rot_xyzw_raw)
    target_rot_xyzw = quat_normalize(target_rot_xyzw_raw)

    root_rot_xyzw = quat_normalize(rot_xyzw[:1] * jp.array([0., 0., 1., 1.]))  # [x, y, z, w] shape (1, 4)

    normalized_rot_xyzw = quat_mul_norm(quat_inverse(root_rot_xyzw), rot_xyzw)
    normalized_pos = quat_rotate(quat_inverse(root_rot_xyzw), normalized_pos)
    normalized_vel = quat_rotate(quat_inverse(root_rot_xyzw), vel)
    normalized_ang = quat_rotate(quat_inverse(root_rot_xyzw), ang)

    normalized_target_rot_xyzw = quat_mul_norm(quat_inverse(root_rot_xyzw), target_rot_xyzw)
    normalized_target_pos = quat_rotate(quat_inverse(root_rot_xyzw), normalized_target_pos)
    normalized_target_vel = quat_rotate(quat_inverse(root_rot_xyzw), target_vel)
    normalized_target_ang = quat_rotate(quat_inverse(root_rot_xyzw), target_ang)

    normalized_rot = normalized_rot_xyzw[:, [3, 0, 1, 2]]
    normalized_target_rot = normalized_target_rot_xyzw[:, [3, 0, 1, 2]]

    qp_normalized = QP(pos=normalized_pos, rot=normalized_rot, vel=normalized_vel, ang=normalized_ang)
    target_qp_normalized = QP(pos=normalized_target_pos, rot=normalized_target_rot, vel=normalized_target_vel, ang=normalized_target_ang)
    return qp_normalized, target_qp_normalized
