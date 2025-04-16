
_base_ = [
    '../_base_/datasets/acdc_night.py',
    '../_base_/ida/ida.py',
    '../_base_/default_runtime.py'
]
checkpoint_path = '/BS/DApt/work/project/segformer_test/pretrained/IDASS_b5.pth'
crop_size = (960, 540)
train_batch = 2
val_batch = 2
lr = 0.0005 
max_iters = 20000
val_interval = 500
val_begin = 0
log_interval = 50

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size = crop_size)
model = dict(
    type='IDASS',
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_path),
    backbone=dict(
        type='MixVisionTransformer_ConvP',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(type='CustomTrainLoop'),
    test_cfg=dict(mode='slide', crop_size=(1024,1024), stride=(768,768)))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=max_iters,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=train_batch, num_workers=4)
val_dataloader = dict(batch_size=val_batch, num_workers=4)
test_dataloader = val_dataloader

# training schedule for 160k
train_cfg = dict(
    type='CustomTrainLoop', max_iters= max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_interval, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, save_best='mIoU', max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

find_unused_parameters=True