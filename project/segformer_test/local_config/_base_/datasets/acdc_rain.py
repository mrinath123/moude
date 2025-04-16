# dataset settings
dataset_type = 'ACDCDataset'
data_root = '/BS/DApt/work/project/segformer_test/acdc/ACDC/'
crop_size = (960, 540)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1920, 1080),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),

    dict(type='PackSegInputs') 
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon_trainvaltest/rgb_anon/rain/train', seg_map_path='gt_trainval/gt/rain/train'), 
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon_trainvaltest/rgb_anon/rain/val', seg_map_path='gt_trainval/gt/rain/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
