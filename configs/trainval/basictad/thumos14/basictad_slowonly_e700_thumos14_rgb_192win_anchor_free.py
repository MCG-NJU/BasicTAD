# 1. data
dataset_type = 'Thumos14Dataset_additional_background'
data_root = './data/thumos14/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_frames=192
img_shape = (160, 160)
overlap_ratio = 0.25

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        video_prefix=data_root + 'frames_6fps/validation',
        pipeline=[
            dict(typename='LoadMetaInfo'),
            dict(typename='LoadAnnotations'),
            dict(typename='Time2Frame'),
            dict(
                typename='TemporalRandomCrop',
                num_frames=num_frames,
                iof_th=0.75),
            dict(typename='LoadFrames', to_float32=True, frame_resize=(180,-1),keep_ratio=True),
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
        video_prefix=data_root + 'frames_6fps/test',
        pipeline=[
            dict(typename='LoadMetaInfo'),
            dict(typename='Time2Frame'),
            dict(
                typename='OverlapCropAug',
                num_frames=num_frames,
                overlap_ratio=overlap_ratio,
                transforms=[
                    dict(typename='TemporalCrop'),
                    dict(typename='LoadFrames', to_float32=True,frame_resize=(180,-1),keep_ratio=True),
                    dict(typename='SpatialCenterCrop', crop_size=img_shape),
                    dict(typename='Normalize', **img_norm_cfg),
                    dict(typename='Pad', size=(num_frames, *img_shape)),
                    dict(typename='DefaultFormatBundle'),
                    dict(typename='Collect', keys=['imgs'])
                ])
        ]))

# 2. model
num_classes = 21
strides = [2,4,8,16,32]
use_sigmoid = False
scales_per_octave = 5
octave_base_scale = 2
num_anchors = scales_per_octave

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='SlowOnly',
        kernel_size=(2,1,1),
        stride=(2,1,1)
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
        typename='FcosHead',
        heads={'act_cls':num_classes, 'offset_reg': 2},
        in_channels=2048,
        num_ins=5,
        num_blocks=3,
        kernel_size=3,
        stride=1,
        padding=1,
    ))

# 3. engines
meshgrid = dict(
    typename='SegmentAnchorMeshGrid',
    strides=strides,
    base_anchor=dict(
        typename='SegmentBaseAnchor',
        base_sizes=strides,
        octave_base_scale=octave_base_scale,
        scales_per_octave=scales_per_octave))

segment_coder = dict(
    typename='DeltaSegmentCoder',
    target_means=[.0, .0],
    target_stds=[1.0, 1.0])

train_engine = dict(
    typename='TrainEngine_AF',
    model=model,
    criterion=dict(
        typename='FcosActFPNContextRegLossCriterion_batches_diou',
        act_loss_type='focal',
        iou_loss_type='iou',
        act_cls=num_classes,
        batch_size=1,
        down_ratio_list=strides,
        num_stacks=5,
        reg_range_list=[[-1., 5.], [2.5, 5.],[2.5, 5.],[2.5, 5.],[2.5,'INF']],
        num_max_acts=60,
        is_thumos=True
        ),
    optimizer=dict(typename='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# 3.2 val engine
val_engine = dict(
    typename='ValEngine_AF',
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename='FcosConverter',
        model=model,
        act_loss_type='focal',
        down_ratio_list=strides,
        act_score_thresh=0.005, 
        norm_offset_reg=True,
        max_proposal=200,
        is_Anet=False,
    ),
    num_classes=num_classes,
    iou_thr=0.6,
    use_sigmoid=use_sigmoid,
    is_Anet=False,
    nmw=True
    )

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
workdir='./workdir/basictad_slowonly_e700_thumos14_rgb_192win_anchor_free'
out='test_slowonly_192win_anchor_free.pkl'