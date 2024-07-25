_base_ = ['./default_runtime.py', './ap10k.py']


scope = 'mmpose'
_scope_ = 'mmpose'
default_scope = 'mmpose'

# runtime
max_epochs = 50
base_lr = 1e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)


log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
find_unused_parameters=False
checkpoint_config = dict(interval=5, create_symlink=False)
evaluation = dict(interval=5, metric='mAP', save_best='AP')

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5)
# total_epochs = 50

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# codec settings
codec = dict(
    type='SimCCLabel', 
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=True,
    use_dark=False)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=None,
)

# model settings
model = dict(
    type='PCT',
    # data preprocessor
    data_preprocessor=dict(
        _scope_='mmpose',
        type='PoseDataPreprocessor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        bgr_to_rgb=True),
    # pretrained='weights/simmim/swin_base.pth',
    # backbone
    backbone=dict(
        type='SwinV2TransformerRPE2FC',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[16, 16, 16, 8],
        pretrain_window_size=[12, 12, 12, 6],
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        rpe_interpolation='geo',
        use_shift=[True, True, False, False],
        relative_coords_table_type='norm8_log_bylayer',
        attn_type='cosine_mh',
        rpe_output_type='sigmoid',
        postnorm=True,
        mlp_type='normal',
        out_indices=(3,),
        patch_embed_type='normal',
        patch_merge_type='normal',
        strid16=False,
        frozen_stages=5,
    ),
    head=dict(
        type='Tokenizer',
        stage_pct='tokenizer',
        in_channels=1024,
        image_size=data_cfg['image_size'],
        num_joints=channel_cfg['num_output_channels'],
        # loss_keypoint=dict(
        #     type='Classifer_loss',
        #     token_loss=1.0,
        #     joint_loss=1.0),
        # cls_head=dict(
        #     conv_num_blocks=2,
        #     conv_channels=256,
        #     dilation=1,
        #     num_blocks=4,
        #     hidden_dim=64,
        #     token_inter_dim=64,
        #     hidden_inter_dim=256,
        #     dropout=0.0),
        tokenizer=dict(
            # guide_ratio=0.0,
            # ckpt="",
            encoder=dict(
                drop_rate=0.2,
                num_blocks=4,
                hidden_dim=512,
                token_inter_dim=64,
                hidden_inter_dim=512,
                dropout=0.0,
            ),
            decoder=dict(
                num_blocks=1,
                hidden_dim=32,
                token_inter_dim=64,
                hidden_inter_dim=64,
                dropout=0.0,
            ),
            codebook=dict(
                token_num=34,
                token_dim=512,
                token_class_num=4096,
                ema_decay=0.9,
            ),
            loss_keypoint=dict(
                type='Tokenizer_loss',
                joint_loss_w=1.0, 
                e_loss_w=15.0,
                beta=0.05,)
            )),
    test_cfg=dict(
        flip_test=False,
        dataset_name='AP10K'))


# base dataset settings
dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = 'data/ap10k/'

backend_args = dict(backend='local')


# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),

    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(
        type='TopdownAffine', 
        input_size=codec['input_size'], 
        use_udp=True
        ),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),


    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        pack_transformed=True
    )
]


val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='TopdownAffine', 
        input_size=codec['input_size'], 
        use_udp=True),
    dict(
        type='PackPoseInputs',
        pack_transformed=True
    )
]

test_pipeline = val_pipeline



# data loaders
data_root = 'data/apt36k'
train_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, 
        data_root=data_root,
        ann_file='annotations/train_annotations_1.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    ))

val_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val_annotations_1.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    ))

test_dataloader = dict(
    batch_size=32,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val_annotations_1.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    ))



# evaluators
val_evaluator = dict(
    type='CocoMetric',
    use_area=True,
    ann_file=f'{data_root}/annotations/val_annotations_1.json')
test_evaluator = dict(
    type='CocoMetric',
    use_area=True,
    ann_file=f'{data_root}/annotations/val_annotations_1.json')

val_cfg = dict()
test_cfg = dict()
