<div align="center">

<h1>InsActor: <br>Instruction-driven Physics-based Characters</h1>

<div>
Jiawei Ren<sup>*</sup>&emsp;Mingyuan Zhang<sup>*</sup>&emsp;Cunjun Yu<sup>*</sup>&emsp;Xiao Ma&emsp;Liang Pan</a>&emsp;Ziwei Liu<sup>&dagger;</sup>
</div>
<div>
    S-Lab, Nanyang Technological University&emsp; 
    National University of Singapore &emsp;<br>
    Dyson Robot Learning Lab &emsp;<br>
    <sup>*</sup>equal contribution <br>
    <sup>&dagger;</sup>corresponding author 
</div>

<div>
   <strong>NeurIPS 2023</strong>
</div>

<div>
<a target="_blank" href="https://arxiv.org/abs/2312.17135">
  <img src="https://img.shields.io/badge/arXiv-2312.17135-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fjiawei-ren%2Finsactor&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>


<div>
<img src="asset/teaser.gif" width="70%"/>
</div>


---

<h4 align="center">
  <a href="https://jiawei-ren.github.io/projects/insactor/index.html" target='_blank'>[Project Page]</a> •
  <a href="https://openreview.net/pdf?id=hXevuspQnX" target='_blank'>[Paper]</a> •
<!-- <a href="https://diffmimic-demo-main-g7h0i8.streamlit.app/" target='_blank'>[Demo]</a> • -->
<a href="https://youtu.be/yej9YINcpvs" target='_blank'>[Video]</a> 

</h4>

</div>

## Installation
```sh
conda create -n insactor python==3.9 -y
conda activate insactor

# diffmimic
python -m pip install --upgrade "jax[cuda]==0.4.2" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# diffplanner
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install -c bottler nvidiacub -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install pytorch3d -c pytorch3d -y

python -m pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)
```

download models
```sh
mkdir pretrained_models && cd pretrained_models
gdown --fuzzy https://drive.google.com/file/d/1qdglrZJa5ago2nkSWc5Nu52kvLaS_dTg/view?usp=drive_link # Human-ML skills 0.001
gdown --fuzzy https://drive.google.com/file/d/1Wi33YZWR8K6IuWZdcXdV2mqcivcyObIU/view?usp=drive_link # Human-ML planner
cd ..
```

## Run Demo
```sh
streamlit run tools/demo.py --server.port 8501
```
The WebUI should be available at `localhost:8501`.

## Run evaluation
```sh
#prepare data
mkdir data/datasets
gdown https://drive.google.com/file/d/1knGb1Kt9VUu377vcXIDfQN_qcAYSYPez/view?usp=drive_link --fuzzy && unzip human_pml3d.zip && rm human_pml3d.zip &&  mv human_pml3d data/datasets/human_pml3d
gdown https://drive.google.com/file/d/1_YQwV5kqZgSHhOJ_V9MwDjKO5yYfsdDP/view?usp=drive_link --fuzzy && unzip kit_pml.zip && rm kit_pml.zip &&  mv kit_pml data/datasets/kit_pml
# prepare contrasitive models
mkdir data/evaluators
gdown https://drive.google.com/file/d/1aoiq702L6fy4yCsX_ub1MxD_mvj-VmrQ/view?usp=drive_link --fuzzy && mv humanml.pth data/evaluators
gdown https://drive.google.com/file/d/1fyGAi2NHHvUNgDN0YgB1ND_7buVel_SA/view?usp=drive_link --fuzzy &&  mv kit.pth data/evaluators
# KIT
export CONTROLLER_PARAM_PATH="pretrained_models/skill_kit_0.01.pkl"
python -u tools/test.py configs/planner/kit.py --work-dir=work_dirs/eval --physmode=normal pretrained_models/planner_kit.pth
# Humanml
export CONTROLLER_PARAM_PATH="pretrained_models/skill_human_0.01.pkl"
python -u tools/test.py configs/planner/human.py --work-dir=work_dirs/eval --physmode=normal pretrained_models/planner_humanml.pth
# Humanml (Perturb)
export CONTROLLER_PARAM_PATH="pretrained_models/skill_human_0.01.pkl"
python -u tools/test.py configs/planner/human.py --work-dir=work_dirs/eval --physmode=normal --perturb true pretrained_models/planner_humanml.pth
# Humanml (Waypoint)
export CONTROLLER_PARAM_PATH="pretrained_models/skill_human_0.01.pkl"
python -u tools/test_waypoint.py configs/planner/human.py --work-dir=work_dirs/eval --physmode=normal pretrained_models/planner_humanml.pth
```
Difference from the paper results:
- We improved the low-level trakcer training by adding a gradient truncation.
- We fixed the contrastive model for humanml-3d.

More models
```sh
cd pretrained_models
gdown --fuzzy https://drive.google.com/file/d/10gFEWUdZtMIA-6yhd_gZXbYm9snMZ63m/view?usp=drive_link # KIT-ML skills 0.001
gdown --fuzzy https://drive.google.com/file/d/1oyT5DE5ItZb1KNSV0lW85cLk9w4alGPV/view?usp=drive_link # Human-ML skills 0.0
gdown --fuzzy https://drive.google.com/file/d/1fvW3RtFU8sOGj2rLZTT7gGSOzLCvNkUW/view?usp=drive_link # KIT-ML skills 0.0
gdown --fuzzy https://drive.google.com/file/d/17ut3gymJpDrPt4nsIhPcug0A0HevqGkE/view?usp=drive_link # Human-ML skills 0.01
gdown --fuzzy https://drive.google.com/file/d/1Ijosu_4W2eIg72kK2IuGaR0wbEhxw6CU/view?usp=drive_link # KIT-ML skills 0.01
gdown --fuzzy https://drive.google.com/file/d/10WrKJ4v1u6DwCYb8KT9s2aP3n2BBn9ST/view?usp=drive_link # KIT-ML planner
cd ..
```

## Visualize Evaluation Results
The evaluation script will save diffusion plans at `planned_traj.npy` and simulated motion in `simulated_traj.npy`.
```sh
streamlit run visualize.py
```
Then input the trajectory file path (e.g., `planned_traj.npy`), or drag and drop the trajectory file. 

## Training Low-level Controllers

```sh
#prepare data
gdown https://drive.google.com/file/d/15qjv-tREix2kJ5kvaRgxTXgoKQgHPq6L/view?usp=drive_link --fuzzy && unzip kit_ml_raw_processed.zip && rm kit_ml_raw_processed.zip &&  mv kit_ml_raw_processed data/kit_ml_raw_processed  # KIT
gdown https://drive.google.com/file/d/1WCiuqQOeIpu2sFjnF20aJiNmYLFMkNxU/view?usp=drive_link --fuzzy && unzip humanml3d_processed.zip && rm humanml3d_processed.zip &&  mv humanml3d_processed data/humanml3d_processed  # Human-ML
# train
python mimic.py --config configs/controller/human_0.01.yaml
```

## Training Diffusion Policy
```sh
python -u tools/train.py configs/planner/kit.py --work-dir=planner_kit  # kit
python -u tools/train.py configs/planner/human.py --work-dir=planner_human  # humanml
```

## Residual Force Control
### RFC with strength 100
Download pretrained policy:
```sh
cd pretrained_models && gdown --fuzzy https://drive.google.com/file/d/1B4FntBB3lFWRgW6uWzZRQWGu0ddPSa2p/view?usp=drive_link && cd ..
```
Then, in (tools/demo_utils/rollout.py)[tools/demo_utils/rollout.py]:
- replace `system_config='smpl` with `system_config='smpl_rfc`
- replace `param_path = 'pretrained_models/skill_human_0.001.pkl'` with `param_path = 'pretrained_models/skill_human_rfc_100.pkl'`

### RFC with strength 1000
Download pretrained policy:
```sh
cd pretrained_models && gdown --fuzzy https://drive.google.com/file/d/1XYfYfFL6sODSlqCpwhddxrh7jX87mkKs/view?usp=drive_link && cd ..
```
Then, in (tools/demo_utils/rollout.py)[tools/demo_utils/rollout.py]:
- replace `system_config='smpl` with `system_config='smpl_rfc_1000`
- replace `param_path = 'pretrained_models/skill_human_0.001.pkl'` with `param_path = 'pretrained_models/skill_human_rfc_1000.pkl'`




## Citation

```
@article{ren2023insactor,
  title={InsActor: Instruction-driven Physics-based Characters},
  author={Ren, Jiawei and Zhang, Mingyuan and Yu, Cunjun and Ma, Xiao and Pan, Liang and Liu, Ziwei},
  journal={NeurIPS},
  year={2023}
}
```
