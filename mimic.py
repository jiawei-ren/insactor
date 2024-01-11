from brax import envs
import functools
import diffmimic.brax_lib.agent_diffmimic as dmm
from absl import flags, app
import yaml
import jax

from diffmimic.utils import AttrDict
from diffmimic.mimic_envs import register_mimic_env
from brax.training.agents.apg import networks as apg_networks

from diffmimic.utils.data import MotionDataset, NumpyLoader, RandomSampler

from brax.io import metrics

register_mimic_env()

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'configs/motion_dataset.yaml', help='Experiment configuration.')


def main(argv):
    with open(FLAGS.config, 'r') as f:
        args = AttrDict(yaml.safe_load(f))

    logdir = "logs/exp"
    pretrain = None
    for k, v in args.items():
        if k == 'motion_dir':
            logdir += f"_{v.split('/')[-1].split('.')[0]}"
        elif k == 'pretrain':
            from brax.io import model
            pretrain = model.load_params(v)
            logdir += f"_resume"
        else:
            logdir += f"_{v}"

    model_fn = functools.partial(
        apg_networks.make_apg_networks,
        hidden_layer_sizes=(512, 512, 512) if not args.get('large', False) else (1024,) * 5
    )

    local_device_count = jax.local_device_count()
    train_set = MotionDataset(motion_dir=args.motion_dir, seq_len=args.ep_len, subset=args.get('subset', None), resample=args.get('resample', False))
    train_sampler = RandomSampler(train_set, replacement=True, num_samples=2**31)
    train_loader = NumpyLoader(
        train_set,
        sampler=train_sampler,
        batch_size=args.num_envs * local_device_count,
        num_workers=0
    )

    test_set = MotionDataset(motion_dir=args.motion_dir, seq_len=args.ep_len_eval, subset=args.get('subset', None))
    test_sampler = RandomSampler(test_set, replacement=True, num_samples=2**31)
    test_loader = NumpyLoader(
        test_set,
        batch_size=args.num_eval_envs,
        sampler=test_sampler,
        num_workers=0
    )
    train_env = envs.get_environment(
        env_name="humanoid_mimic_train",
        system_config=args.system_config,
        early_termination=args.early_termination,
        demo_replay_mode=args.demo_replay_mode,
        err_threshold=args.threshold,
        replay_rate=args.replay_rate,
        reward_scaling=args.reward_scaling,
        vel_weight=args.vel_weight,
        rot_weight=args.get('rot_weight', 0.25),
        ang_weight=args.get('ang_weight', 0.01),
        foot_weight=args.get('foot_weight', 0.0),
        local=args.get('local', 'none'),
    )

    eval_env = envs.get_environment(
        env_name="humanoid_mimic",
        system_config=args.system_config,
        vel_weight=args.vel_weight,
        rot_weight=args.get('rot_weight', 0.25),
        ang_weight=args.get('ang_weight', 0.01),
        foot_weight=args.get('foot_weight', 0.0),
        local=args.get('local', 'none'),
    )

    with metrics.Writer(logdir) as writer:
        make_inference_fn, params, _ = dmm.train(
            seed=args.seed,
            environment=train_env,
            eval_environment=eval_env,
            episode_length=args.ep_len-1,
            eval_episode_length=args.ep_len_eval-1,
            num_envs=args.num_envs,
            num_eval_envs=args.num_eval_envs,
            learning_rate=args.lr,
            num_evals=args.max_it+1,
            max_gradient_norm=args.max_grad_norm,
            network_factory=model_fn,
            normalize_observations=args.normalize_observations,
            save_dir=logdir,
            progress_fn=writer.write_scalars,
            use_linear_scheduler=args.use_lr_scheduler,
            truncation_length=args.truncation_length,
            train_loader=train_loader,
            test_loader=test_loader,
            pretrain=pretrain,
            beta=args.beta,
            deterministic=args.get('deterministic', False),
            large=args.get('large', False),
            conditional=args.get('conditional', False),
            weight_decay=args.get('weight_decay', 0),
            skip_encoder=args.get('skip_encoder', False),
        )


if __name__ == '__main__':
    app.run(main)
