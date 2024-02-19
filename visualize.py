import numpy as np
import os
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import streamlit as st
from diffmimic.utils.io import deserialize_qp, serialize_qp
from simulate.humanoid_mimic_hit import HumanoidMimic as HumanoidMimicHit
envs.register_environment('humanoid_mimic_hit', HumanoidMimicHit)

from diffmimic.mimic_envs import register_mimic_env

register_mimic_env()

st.title('InsActor - Visualization')
with st.expander("Readme"):
    st.markdown(
        'Input modes:'
        '\n- File Path: input the local file path to the .npy trajectory file.'
        '\n- Direct Upload: directly upload the .npy trajectory file.')


def show_rollout_traj(rollout_traj, tag):
    if len(rollout_traj.shape) == 3:
        seed = st.slider(f'Random seed ({tag})', 0, rollout_traj.shape[1] - 1, 0)
        rollout_traj = rollout_traj[:, seed]

    if rollout_traj.shape[-1] > 247:
        env_name = 'humanoid_mimic_hit'
    else:
        env_name = 'humanoid_mimic'

    rollout_qp = [deserialize_qp(rollout_traj[i]) for i in range(rollout_traj.shape[0])]
    rollout_traj = serialize_qp(deserialize_qp(rollout_traj))

    env = envs.get_environment(
        env_name=env_name,
        system_config='smpl',
    )
    components.html(html.render(env.sys, rollout_qp, height=500), height=500)


def main():
    tab_file_path, tab_direct_upload = st.tabs(["File Path", "Direct Upload"])

    with tab_file_path:
        file_path = st.text_input('Path to trajectory')
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                rollout_traj = np.load(f)
                show_rollout_traj(rollout_traj, 'FP')
        else:
            st.warning('Please input a valid path', icon="⚠️")

    with tab_direct_upload:
        uploaded_file = st.file_uploader("Upload a trajectory")
        if uploaded_file is not None:
            rollout_traj = np.load(uploaded_file)
            show_rollout_traj(rollout_traj, 'DU')


if __name__ == '__main__':
    main()
