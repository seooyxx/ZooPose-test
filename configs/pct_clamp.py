_base_ = ['./default_runtime.py', './ap10k.py']

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys={'text_encoder': dict(lr_mult=0.0),
                                    'backbone': dict(lr_mult=0.1),
                                    'norm': dict(decay_mult=0.)})
)

codec = dict(
    type='UDPHeatmap', input_size=(256, 256), heatmap_size=(56, 56), sigma=2
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200]
)
total_epochs = 210
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ]
)

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
    bbox_file='',
)

# model settings
model = dict(
    type='PCT',
    pretrained='weights/heatmap/swin_heatmap_best_AP_epoch_65.pth',
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
    keypoint_head=dict(
        type='PCT_Clamp_Head',
        stage_pct='classifier',
        in_channels=1024,
        out_channels=17,
        image_size=data_cfg['image_size'],
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(
            type='CompositeLoss',  # Updated loss to combine original losses with CLAMP loss
            token_loss=1.0,
            joint_loss=1.0,
            clamp_loss_weight=1.0,  # Weight for the CLAMP contrastive loss
            clamp_loss_type='CE',  # Choose between 'CE' (Cross-Entropy) or 'NCE'
            text_embedding_dim=512,
            visual_embedding_dim=1024,
        ),
        cls_head=dict(
            conv_num_blocks=2,
            conv_channels=256,
            dilation=1,
            num_blocks=4,
            hidden_dim=64,
            token_inter_dim=64,
            hidden_inter_dim=256,
            dropout=0.0),
        tokenizer=dict(
            guide_ratio=0.5,
            ckpt="weights/tokenizer/epoch_50.pth",
            encoder=dict(
                drop_rate=0.4,  # default = 0.2
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
                token_class_num=4096,  # default: 2048
                ema_decay=0.9,
            ),
            loss_keypoint=dict(
                type='Tokenizer_loss',
                joint_loss_w=1.0, 
                e_loss_w=15.0,
                beta=0.05,)
        ),
        text_encoder=dict(
            type='CLIPTextContextEncoder',
            context_length=13,
            embed_dim=512,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            pretrained='weights/clip/ViT-B-16.pt',
            style='pytorch'
        ),
        visual_encoder=dict(
            type='CLIPVisionTransformer',
            embed_dim=1024,
            transformer_width=768,
            transformer_heads=12,
            transformer_layers=12,
            pretrained='weights/clip/ViT-B-16.pt',
            style='pytorch'
        ),
        pose_encoder=dict(
            type='PoseEncoderModule',  # 추가된 Pose Encoder
            input_dim=17 * 3,  # Assuming 17 joints, each with (x, y, visibility)
            output_dim=512,    # Output dimension to match with visual and text embeddings
            num_layers=3,
            hidden_dim=1024,
        )
    ),
    test_cfg=dict(
        flip_test=True,
        dataset_name='AP10K'
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
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
                p=1.),
        ]
    ),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs', pack_transformed=True)
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='TopdownAffine', 
        input_size=codec['input_size'], 
        use_udp=True
    ),
    dict(type='PackPoseInputs', pack_transformed=True)
]

test_pipeline = val_pipeline

dataset_type = 'AP10KDataset'
data_mode = 'topdown'

# data loaders
data_root = 'data/APTv2'
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type, 
        data_root=data_root,
        ann_file='annotations/train_annotations.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val_annotations.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    )
)

test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test_annotations.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/ap10k.py')
    )
)

val_evaluator = dict(
    type='CocoMetric',
    use_area=True,
    ann_file=f'{data_root}/annotations/val_annotations.json'
)
test_evaluator = dict(
    type='CocoMetric',
    use_area=True,
    ann_file=f'{data_root}/annotations/val_annotations.json'
)

val_cfg = dict()
test_cfg = dict()
