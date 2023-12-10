_base_ = ['../_base_/datasets/human_pml3d_ttc_bs32_2t4m.py']

# checkpoint saving
checkpoint_config = dict(interval=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# optimizer
optimizer = dict(type='Adam', lr=2e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=80)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

input_feats = 247
max_seq_len = 196
latent_dim = 512
time_embed_dim = 2048
text_latent_dim = 256
ff_size = 1024
num_heads = 8
dropout = 0
# model settings
model = dict(
    type='MotionDiffusion',
    model=dict(
        type='ReMoDiffuseTransformer',
        input_feats=input_feats,
        max_seq_len=max_seq_len,
        latent_dim=latent_dim,
        time_embed_dim=time_embed_dim,
        num_layers=4,
        ca_block_cfg=dict(
            type='SemanticsModulatedAttention',
            latent_dim=latent_dim,
            text_latent_dim=text_latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            time_embed_dim=time_embed_dim
        ),
        ffn_cfg=dict(
            latent_dim=latent_dim,
            ffn_dim=ff_size,
            dropout=dropout,
            time_embed_dim=time_embed_dim
        ),
        text_encoder=dict(
            pretrained_model='clip',
            latent_dim=text_latent_dim,
            num_layers=4,
            ff_size=2048,
            dropout=dropout,
            use_text_proj=False
        ),
        guide_scale=2.5
    ),
    loss_recon=dict(type='MSELoss', loss_weight=1, reduction='none'),
    diffusion_train=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_small',
    ),
    diffusion_test=dict(
        beta_scheduler='linear',
        diffusion_steps=1000,
        model_mean_type='start_x',
        model_var_type='fixed_small',
        respace='15,15,8,6,6',
    ),
    inference_type='ddim'
)
