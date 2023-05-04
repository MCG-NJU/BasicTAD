# 1. data
dataset_type = 'Thumos14Dataset_additional_background'
data_root = './data/thumos14/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_frames=96
img_shape = (112,112)
img_shape_test = (128,128)
overlap_ratio = 0.25

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        video_prefix=data_root + 'frames_3fps/validation', 
        pipeline=[
            dict(typename='LoadMetaInfo'),
            dict(typename='LoadAnnotations'),
            dict(typename='Time2Frame'),
            dict(
                typename='TemporalRandomCrop',
                num_frames=num_frames,
                iof_th=0.75),
            dict(typename='LoadFrames', to_float32=True, frame_resize=(128,-1),keep_ratio=True),
            dict(typename='SpatialRandomCrop', crop_size=img_shape),
            dict(
                typename='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
                p=0.5),
            dict(
                typename='Rotate',
                limit=(-45, 45),
                border_mode='reflect101',
                p=0.5),
            dict(typename='SpatialRandomFlip', flip_ratio=0.5),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='Pad', size=(num_frames, *img_shape)),
            dict(typename='DefaultFormatBundle'),
            dict(
                typename='Collect',
                keys=[
                    'imgs', 'gt_segments', 'gt_labels', 'gt_segments_ignore'
                ])
        ]),
    val=dict(
        typename=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        video_prefix=data_root + 'frames_3fps/test',
        pipeline=[
            dict(typename='LoadMetaInfo'),
            dict(typename='Time2Frame'),
            dict(
                typename='OverlapCropAug',
                num_frames=num_frames,
                overlap_ratio=overlap_ratio,
                transforms=[
                    dict(typename='TemporalCrop'),
                    dict(typename='LoadFrames', to_float32=True,frame_resize=(128,-1),keep_ratio=True),
                    dict(typename='SpatialCenterCrop', crop_size=img_shape_test),
                    dict(typename='Normalize', **img_norm_cfg),
                    dict(typename='Pad', size=(num_frames, *img_shape_test)),
                    dict(typename='DefaultFormatBundle'),
                    dict(typename='Collect', keys=['imgs'])
                ])
        ]))

# 2. model
num_classes = 20
strides = [1,2,4,8,16]
use_sigmoid = True
scales_per_octave = 5
octave_base_scale = 2
num_anchors = scales_per_octave

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='SlowOnly',
    ),
    neck=[
        dict(
            typename='AF_tdm', 
            srm_cfg=dict(
                typename='AdaptiveAvgPool3d', output_size=(None, 1, 1)
            ),
            num_layers=5,
            kernel_size=3,
            stride=2,
            padding=1
            )
    ],
    head=dict(
        typename='FcosHead_sigmoid',
        num_classes=num_classes,
        in_channels=2048,
        stacked_convs=4,
        dcn_on_last_conv=True,
        feat_channels=2048,
        use_sigmoid=use_sigmoid,
        num_ins=5,
        conv_cfg=dict(typename='Conv1d'),
        norm_cfg=dict(typename='GN',num_groups=32, requires_grad=True),
    ))

# 3. engines
meshgrid = dict(
    typename='PointAnchorMeshGrid',
    strides=strides)

segment_coder = dict(
    typename='DeltaSegmentCoder',
    target_means=[.0, .0],
    target_stds=[1.0, 1.0])

train_engine = dict(
    typename='TrainEngine_AF',
    model=model,
    criterion=dict(
        typename='FcosCriterion',
        reg_range_list=[[-1., 5.], [2.5, 5.],[2.5, 5.],[2.5, 5.],[2.5,'INF']],
        loss_cls=dict(
            typename='FocalLoss',
            use_sigmoid=use_sigmoid,
            change_background=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            num_classes=num_classes,
            ),
        loss_segment=dict(typename='DIoULoss', loss_weight=1.0),
        num_classes=num_classes,
        strides = strides,
        is_thumos = True,
        ),
    optimizer=dict(typename='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# 3.2 val engine
val_engine = dict(
    typename='ValEngine',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='PointAnchorConverter',
        down_ratio_list=strides,
        num_classes=num_classes,
        segment_coder=segment_coder,
        nms_pre=1000,
        use_sigmoid=use_sigmoid),
    num_classes=num_classes,
    test_cfg=dict(
        score_thr=0.005,
        nms=dict(typename='nmw-af', iou_thr=0.4),
        max_per_video=1200), #1200
    use_sigmoid=use_sigmoid)

# 4. hooks
hooks = [
    dict(typename='OptimizerHook'),
    dict(
        typename='CosineRestartLrSchedulerHook',
        periods=[100] * 12,
        restart_weights=[1] * 12,
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1e-1,
        min_lr_ratio=1e-2),
    dict(typename='EvalHook', eval_cfg=dict(mode='anet')),
    dict(typename='SnapshotHook', interval=100),
    dict(typename='LoggerHook', interval=10)
]

# 5. work modes
modes = ['train']
max_epochs = 1200

# 6. misc
seed = 10
dist_params = dict(backend='nccl')
log_level = 'INFO'
find_unused_parameters = True
deterministic = True
workdir='./workdir/basictad_slowonly_e700_thumos14_rgb_96win_anchor_free'
out='test_slowonly_96win_anchor_free.pkl'
