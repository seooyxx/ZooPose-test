# --------------------------------------------------------
# Pose Compositional Tokens
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.builder import LOSSES

class CompositeLoss(nn.Module):
    def __init__(self, token_loss=1.0, joint_loss=1.0, clamp_loss_weight=1.0, clamp_loss_type='CE', text_embedding_dim=512, visual_embedding_dim=1024):
        super(CompositeLoss, self).__init__()
        self.token_loss_weight = token_loss
        self.joint_loss_weight = joint_loss
        self.clamp_loss_weight = clamp_loss_weight
        self.clamp_loss_type = clamp_loss_type
        
        self.token_loss_fn = nn.CrossEntropyLoss()  # or another appropriate loss for tokens
        self.joint_loss_fn = nn.MSELoss()  # Mean Squared Error Loss for joint predictions
        
        # Projection layers for contrastive learning
        self.text_projection = nn.Linear(text_embedding_dim, visual_embedding_dim)
        self.visual_projection = nn.Linear(visual_embedding_dim, visual_embedding_dim)
        
        if self.clamp_loss_type == 'CE':
            self.clamp_loss_fn = nn.CrossEntropyLoss()
        elif self.clamp_loss_type == 'NCE':
            self.clamp_loss_fn = self.noise_contrastive_estimation

    def forward(self, token_preds, joint_preds, token_labels, joint_labels, text_embeddings, visual_features):
        # Token loss
        token_loss = self.token_loss_fn(token_preds, token_labels)
        
        # Joint loss
        joint_loss = self.joint_loss_fn(joint_preds, joint_labels)
        
        # CLAMP contrastive loss
        text_proj = self.text_projection(text_embeddings)
        visual_proj = self.visual_projection(visual_features)
        
        if self.clamp_loss_type == 'CE':
            contrastive_loss = self.clamp_loss_fn(text_proj, visual_proj)
        elif self.clamp_loss_type == 'NCE':
            contrastive_loss = self.clamp_loss_fn(text_proj, visual_proj)
        
        # Total loss
        total_loss = (self.token_loss_weight * token_loss +
                      self.joint_loss_weight * joint_loss +
                      self.clamp_loss_weight * contrastive_loss)
        
        return total_loss
    
    def noise_contrastive_estimation(self, text_proj, visual_proj):
        # Implement NCE loss if needed
        logits = torch.matmul(text_proj, visual_proj.t())
        labels = torch.arange(logits.size(0)).to(logits.device)
        nce_loss = F.cross_entropy(logits, labels)
        return nce_loss