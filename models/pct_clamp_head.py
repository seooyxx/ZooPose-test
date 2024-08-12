import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import constant_init, normal_init
from mmpose.models.builder import build_loss, HEADS
from mmpose.models.heads.base_head import BaseHead

from .pct_tokenizer import Tokenizer
from .modules import MixerLayer, FCBlock, BasicBlock


@HEADS.register_module()
class PCT_Clamp_Head(nn.Module):
    def __init__(self,
                 stage_pct,
                 in_channels,
                 out_channels,
                 image_size,
                 num_joints,
                 cls_head=None,
                 tokenizer=None,
                 loss_keypoint=None,
                 text_encoder=None,       
                 visual_encoder=None,     
                 pose_encoder=None):      
        super().__init__()

        self.image_size = image_size
        self.stage_pct = stage_pct

        self.text_encoder = text_encoder
        self.visual_encoder = visual_encoder
        self.pose_encoder = pose_encoder  

        self.guide_ratio = tokenizer['guide_ratio']
        self.img_guide = self.guide_ratio > 0.0

        self.conv_channels = cls_head['conv_channels']
        self.hidden_dim = cls_head['hidden_dim']

        self.num_blocks = cls_head['num_blocks']
        self.hidden_inter_dim = cls_head['hidden_inter_dim']
        self.token_inter_dim = cls_head['token_inter_dim']
        self.dropout = cls_head['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']

        if stage_pct == "classifier":
            self.conv_trans = self._make_transition_for_head(
                in_channels, self.conv_channels)
            self.conv_head = self._make_cls_head(cls_head)

            input_size = (image_size[0]//32)*(image_size[1]//32)
            self.mixer_trans = FCBlock(
                self.conv_channels * input_size, 
                self.token_num * self.hidden_dim)

            self.mixer_head = nn.ModuleList(
                [MixerLayer(self.hidden_dim, self.hidden_inter_dim,
                    self.token_num, self.token_inter_dim,  
                    self.dropout) for _ in range(self.num_blocks)])
            self.mixer_norm_layer = FCBlock(
                self.hidden_dim, self.hidden_dim)

            self.cls_pred_layer = nn.Linear(
                self.hidden_dim, self.token_class_num)
        
        self.tokenizer = Tokenizer(
            stage_pct=stage_pct, tokenizer=tokenizer, num_joints=num_joints)

        self.loss = build_loss(loss_keypoint)
        print(f'loss: {self.loss}')

    def get_loss(self, p_logits, p_joints, g_logits, joints, text_embeddings, visual_features, pose_embeddings):
        """Calculate loss for training classifier, with contrastive learning."""

        losses = dict()

        # 기존 손실 계산
        losses['token_loss'], losses['kpt_loss'] = self.loss(p_logits, p_joints, g_logits, joints)

        # 텍스트-이미지 대조 학습 손실 추가 (Cross-Entropy 또는 NCE)
        text_proj = self.text_encoder(text_embeddings)
        visual_proj = self.visual_encoder(visual_features)
        contrastive_loss_text = F.cross_entropy(text_proj, visual_proj)
        losses['contrastive_loss_text'] = contrastive_loss_text

        # 포즈-이미지 대조 학습 손실 추가
        pose_proj = self.pose_encoder(joints)  # 포즈를 포즈 인코더로 인코딩
        contrastive_loss_pose = F.cross_entropy(pose_proj, visual_proj)
        losses['contrastive_loss_pose'] = contrastive_loss_pose

        unused_losses = []
        for name, loss in losses.items():
            if loss is None:
                unused_losses.append(name)
        for unused_loss in unused_losses:
            losses.pop(unused_loss)

        return losses

    def forward(self, x, extra_x, joints=None, texts=None, train=True):
        """Forward function."""

        if self.stage_pct == "classifier":
            batch_size = x[-1].shape[0]
            cls_feat = self.conv_head[0](self.conv_trans(x[-1]))

            cls_feat = cls_feat.flatten(2).transpose(2,1).flatten(1)
            cls_feat = self.mixer_trans(cls_feat)
            cls_feat = cls_feat.reshape(batch_size, self.token_num, -1)

            for mixer_layer in self.mixer_head:
                cls_feat = mixer_layer(cls_feat)
            cls_feat = self.mixer_norm_layer(cls_feat)

            cls_logits = self.cls_pred_layer(cls_feat)

            encoding_scores = cls_logits.topk(1, dim=2)[0]
            cls_logits = cls_logits.flatten(0,1)
            cls_logits_softmax = cls_logits.clone().softmax(1)
        else:
            encoding_scores = None
            cls_logits = None
            cls_logits_softmax = None

        if not self.img_guide or (self.stage_pct == "classifier" and not train):
            joints_feat = None
        else:
            joints_feat = None

        # 텍스트 및 이미지 임베딩 추출
        if texts is not None:
            text_embeddings = self.text_encoder(texts)
        else:
            text_embeddings = None

        visual_features = cls_feat

        # 포즈 임베딩 추출
        pose_embeddings = self.pose_encoder(joints)

        output_joints, cls_label, e_latent_loss = \
            self.tokenizer(joints, joints_feat, cls_logits_softmax, train)

        if train:
            losses = self.get_loss(cls_logits, output_joints, cls_label, joints, text_embeddings, visual_features, pose_embeddings)
            return cls_logits, output_joints, cls_label, e_latent_loss, losses
        else:
            return output_joints, encoding_scores

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_cls_head(self, layer_config):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            layer_config['conv_channels'],
            layer_config['conv_channels'],
            layer_config['conv_num_blocks'],
            dilation=layer_config['dilation']
        )
        feature_convs.append(feature_conv)
        
        return nn.ModuleList(feature_convs)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def extract_joints_feat(self, feature_map, joint_coords):
        assert self.image_size[1] == self.image_size[0], \
            'If you want to use a rectangle input, ' \
            'please carefully check the length and width below.'
        batch_size, _, _, height = feature_map.shape
        stride = self.image_size[0] / feature_map.shape[-1]
        joint_x = (joint_coords[:,:,0] / stride + 0.5).int()
        joint_y = (joint_coords[:,:,1] / stride + 0.5).int()
        joint_x = joint_x.clamp(0, feature_map.shape[-1] - 1)
        joint_y = joint_y.clamp(0, feature_map.shape[-2] - 1)
        joint_indices = (joint_y * height + joint_x).long()

        flattened_feature_map = feature_map.clone().flatten(2)
        joint_features = flattened_feature_map[
            torch.arange(batch_size).unsqueeze(1), :, joint_indices]

        return joint_features

    def init_weights(self):
        if self.stage_pct == "classifier":
            self.tokenizer.eval()
            for name, params in self.tokenizer.named_parameters():
                params.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
