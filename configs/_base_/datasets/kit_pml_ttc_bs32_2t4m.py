# dataset settings
data_keys = ['motion', 'motion_mask', 'motion_length']
meta_keys = ['text', 'token']
train_pipeline = [
    dict(type='Crop', crop_size=196),
    dict(
        type='Normalize',
        mean_path='data/datasets/kit_pml/mean.npy',
        std_path='data/datasets/kit_pml/std.npy'),
    dict(type='ToTensor', keys=data_keys),
    dict(type='Collect', keys=data_keys, meta_keys=meta_keys)
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        dataset=dict(
            type='TextMotionDataset',
            dataset_name='kit_pml',
            data_prefix='data',
            pipeline=train_pipeline,
            ann_file='train.txt',
            motion_dir='motions',
            text_dir='texts',
            token_dir='tokens',
        ),
        times=50
    ),
    test=dict(
        type='TextMotionDataset',
        dataset_name='kit_pml',
        data_prefix='data',
        pipeline=train_pipeline,
        ann_file='test.txt',
        motion_dir='motions',
        text_dir='texts',
        token_dir='tokens',
        eval_cfg=dict(
            shuffle_indexes=True,
            replication_times=3,
            replication_reduction='statistics',
            text_encoder_name='kit_ttc',
            text_encoder_path='data/evaluators/kit.pth',
            motion_encoder_name='kit_ttc',
            motion_encoder_path='data/evaluators/kit.pth',
            metrics=[
                dict(type='R Precision', batch_size=32, top_k=3),
                dict(type='Matching Score', batch_size=32),
                dict(type='FID'),
                dict(type='Diversity', num_samples=300),
            ]
        ),
        test_mode=True
    )
)