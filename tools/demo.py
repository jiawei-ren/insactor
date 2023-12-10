import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import sys
from diffmimic.utils.io import deserialize_qp, serialize_qp
import streamlit as st
# st.set_page_config(layout="wide")
with st.spinner('Loading Diffusion Model'):
    from demo_utils.diffuse import infer_motion_diffusion
with st.spinner('Loading Generative Controller'):
    from demo_utils.rollout import execute_actions
from demo_utils.scene import add_scene_to_traj, remove_scene_from_traj

from simulate.humanoid_mimic_hit import HumanoidMimic as HumanoidMimicHit
envs.register_environment('humanoid_mimic_hit', HumanoidMimicHit)

st.title('InsActor - Demo')

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


def display(traj, history):
    if history is not None:
        traj = np.concatenate([history[:, :-1], traj], axis=1)
    env = envs.get_environment(
        env_name="humanoid_mimic_hit",
        system_config='smpl',
    )
    traj = traj.transpose(1, 0, 2)  # TxNxD
    num_row, num_col = 2, 2
    seed = 0
    for _ in range(num_row):
        for col in st.columns(num_col):
            with col:
                qp = [deserialize_qp(traj[i, seed]) for i in range(traj.shape[0])]
                components.html(html.render(env.sys, qp, height=300), height=300)
                seed += 1
    # qp = [deserialize_qp(traj[i, 0]) for i in range(traj.shape[0])]
    # components.html(html.render(env.sys, qp, height=300), height=300)

def set_endpoint_to_zero(data):
    num_obj = data.shape[-1] // 13
    end_loc = np.zeros([data.shape[0], data.shape[1], num_obj, 3])
    end_loc[..., :18, :2] += np.copy(data[:, -1:, None, :2])
    end_loc[..., 19:, :2] += np.copy(data[:, -1:, None, :2])
    end_loc = end_loc.reshape([data.shape[0], data.shape[1], num_obj * 3])
    data[..., :num_obj*3] = np.copy(data[..., :num_obj*3]) - end_loc
    return  data

def set_endpoint_to_xy(data, x, y):
    num_obj = data.shape[-1] // 13
    end_loc = np.zeros([data.shape[0], data.shape[1], num_obj, 3])
    end_loc[..., :18, 0] += x
    end_loc[..., 19:, 0] += x
    end_loc[..., :18, 1] += y
    end_loc[..., 19:, 1] += y
    end_loc = end_loc.reshape([data.shape[0], data.shape[1], num_obj * 3])
    data[..., :num_obj*3] = np.copy(data[..., :num_obj*3]) + end_loc
    return  data

def set_startpoint_to_zero(data):
    num_obj = data.shape[-1] // 13
    end_loc = np.zeros([data.shape[0], data.shape[1], num_obj, 3])
    end_loc[..., :18, :2] += np.copy(data[:, :1, None, :2])
    end_loc[..., 19:, :2] += np.copy(data[:, :1, None, :2])
    end_loc = end_loc.reshape([data.shape[0], data.shape[1], num_obj * 3])
    data[..., :num_obj*3] = np.copy(data[..., :num_obj*3]) - end_loc
    return  data

def diffusion_planner(text, pre_seq, waypoint, motion_length):
    if waypoint is not None and pre_seq is not None:
        pre_seq = set_endpoint_to_xy(pre_seq, -0.5 * waypoint[0], -0.5 * waypoint[1])
    pred_motion = infer_motion_diffusion(text, pre_seq, waypoint, motion_length)
    if pre_seq is not None:
        pred_motion = set_startpoint_to_zero(pred_motion[:, pre_seq.shape[1]-1:])
        pre_seq = set_endpoint_to_zero(pre_seq)
        pred_motion = np.concatenate([pre_seq[:, -1:], pred_motion[:, 1:]], axis=1) # include the start state
    elif waypoint is not None:
        pred_motion = set_startpoint_to_zero(pred_motion)
    return np.array(pred_motion)

def generative_controller(planned_motion, perturb, motion_length):
    if motion_length == planned_motion.shape[1]:
        rollout_traj = execute_actions(planned_motion, perturb)
        rollout_traj = rollout_traj[:, :-1]
        return np.array(rollout_traj)
    else:
        padded_planned_motion = np.concatenate([planned_motion, planned_motion[:, :motion_length-planned_motion.shape[1]]], 1)
        rollout_traj = execute_actions(padded_planned_motion, perturb)
        rollout_traj = rollout_traj[:, :planned_motion.shape[1]]
        return np.array(rollout_traj)

if __name__ == '__main__':
    with st.form(key='text_input'):
        text = st.text_input(label='Motion description', value="A person is walking like a zombie.")
        ready_to_generate = st.form_submit_button(label='Generate')
        with st.expander('More Control'):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                motion_length = st.number_input('Motion Length (s)', value=6, min_value=1, max_value=10000)
                autoregressive = st.checkbox(label="Keep History", value=False)
                show_plan = st.checkbox(label="Show Plan", value=True)
                show_sim = st.checkbox(label="Show Simulated", value=True)
            with col2:
                perturb = st.checkbox(label="Perturb", value=False)
            with col3:
                apply_target = st.checkbox(label="Set Target", value=False)
                f_ed = st.number_input('Forward Dir ', value=3., min_value=-5., max_value=5., step=0.5)
                r_ed = st.number_input('Right Dir.', value=0., min_value=-5., max_value=5., step=0.5)
                y_ed = f_ed * -1
                x_ed = r_ed * -1

    
    if ready_to_generate:
        if 'history' in st.session_state and autoregressive:
            pre_seq = remove_scene_from_traj(st.session_state.history)[0][:, -60:]
            pre_seq = np.array(pre_seq)
            history = st.session_state.history
        else:
            pre_seq = None
            history = None
        x_ed = x_ed if apply_target else None
        y_ed = y_ed if apply_target else None
        waypoint = (x_ed, y_ed) if apply_target else None
        if waypoint is None:
            scene = 'vis'
        else:
            scene = 'hit'
        with st.spinner('Running high-level diffusion planner'):
            planned_motion = diffusion_planner(text, pre_seq, waypoint, motion_length * 30)
            planned_motion = add_scene_to_traj(planned_motion, waypoint, scene, bar_height=1.4)
        if show_plan:
            st.write('Planned Motion:')
            display(planned_motion, history)

        if show_sim:
            st.write('Simulated Motion:')
            with st.spinner('Running low-level skill mapping (first run may take a minute)'):
                rollout_traj = generative_controller(planned_motion, perturb=perturb, motion_length=motion_length * 30)
            # st.write('Simulated Animation:')
            display(rollout_traj, history)
            if autoregressive:
                if 'history' in st.session_state:
                    rollout_traj = np.concatenate([st.session_state.history[:, :-1], rollout_traj], axis=1)
                    del st.session_state['history']
                rollout_traj = np.array(rollout_traj)
                st.session_state.history = set_endpoint_to_zero(rollout_traj)
            else:
                if 'history' in st.session_state:
                    del st.session_state['history']