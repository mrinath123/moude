# dataset settings
dataset_type = 'SynDataset'
data_root = 'data/'
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
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), #dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs') 
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),  # Load annotations before resizing
    dict(type='Resize', scale=(1024, 512), keep_ratio=False),

    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='weather_cityscapes/weather_cityscapes/leftImg8bit/train/rain/100mm/rainy_image', seg_map_path='cityscapes/gtFine/train'), 
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
            img_path='weather_cityscapes/weather_cityscapes/leftImg8bit/val/rain/100mm/rainy_image', seg_map_path='cityscapes/gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator