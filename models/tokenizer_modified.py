import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmpose.models.builder import build_loss
from mmpose.models.builder import HEADS
from timm.models.layers import trunc_normal_

from .modules import MixerLayer



class CodebookDecoder(nn.Module):
    ''' Codebook & Decoder Module for Tokenizer. '''
    def __init__(self, tokenizer=None, num_joints=17):
        super(CodebookDecoder, self).__init__()
        
        self.num_joints = num_joints
        self.token_num = tokenizer['codebook']['token_num']
        self.token_dim = tokenizer['codebook']['token_dim']

        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']
        self.token_dim = tokenizer['codebook']['token_dim']
        self.decay = tokenizer['codebook']['ema_decay']

        self.token_mlp = nn.Linear(self.token_num, self.num_joints)
        self.decoder_start = nn.Linear(self.token_dim, self.dec_hidden_dim)

        # self.layers
        self.decoder_layers = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)

        ###### 

        self.register_buffer('codebook', torch.empty(self.token_class_num, self.token_dim))
        self.codebook.data.normal_()

        # decoder.ema_cluster_size # decoder.ema_w
        self.register_buffer('ema_cluster_size', torch.zeros(self.token_class_num))
        self.register_buffer('ema_w', torch.empty(self.token_class_num, self.token_dim))
        self.ema_w.data.normal_()  


    def forward(self, encode_feat, device, bs):

        # need: encode_feat
        # codebook
        distances = torch.sum(encode_feat**2, dim=1, keepdim=True) \
            + torch.sum(self.codebook**2, dim=1) \
            - 2 * torch.matmul(encode_feat, self.codebook.t())
        
        # find the closest encoding indicies
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.token_class_num, device=device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        part_token_feat = torch.matmul(encodings, self.codebook)
        part_token_feat = encode_feat + (part_token_feat - encode_feat).detach()

        # return part_token_feat, encodings, encoding_indices
        original_part_token_feat = part_token_feat.clone()

        # decoding part, recover joints
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder_layers:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.layer_norm(decode_feat)

        recoverd_joints = self.recover_embed(decode_feat)

        # return recovered joints
        return original_part_token_feat, encodings, \
            encoding_indices, recoverd_joints


    def codebook_update(self, encodings, encode_feat, part_token_feat):
        """ update codebook using EMA update """
        
        dw = torch.matmul(encodings.t(), encode_feat.detach())
        
        # sync across all process
        n_encodings, n_dw = encodings.numel(), dw.numel()
        encodings_shape, dw_shape = encodings.shape, dw.shape
        combined = torch.cat((encodings.flatten(), dw.flatten()))

        dist.all_reduce(combined) # math sum

        sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
        sync_encodings, sync_dw = \
                sync_encodings.view(encodings_shape), sync_dw.view(dw_shape)
        
        # update the EMA cluster size and codebook
        self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                (1 - self.decay) * torch.sum(sync_encodings, 0)
        
        n = torch.sum(self.ema_cluster_size.data)
        self.ema_cluster_size = ((self.ema_cluster_size + 1e-5) / (n + self.token_class_num * 1e-5) * n)

        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * sync_dw
        self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(1)
        
        # calculate the loss
        e_latent_loss = F.mse_loss(part_token_feat.detach(), encode_feat)

        return e_latent_loss


@HEADS.register_module()
class Tokenizer(nn.Module):
    """ Tokenizer of Pose Compositional Tokens.
        paper ref: Zigang Geng et al. "Human Pose as
            Compositional Tokens"

    Args:
        stage_pct (str): Training stage (Tokenizer or Classifier).
        tokenizer (list): Config about the tokenizer.
        num_joints (int): Number of annotated joints in the dataset.
        guide_ratio (float): The ratio of image guidance.
        guide_channels (int): Feature Dim of the image guidance.
    """

    def __init__(self,
                 stage_pct,
                 num_joints=17,  
                 tokenizer=None,
                 ):
        super().__init__()

        self.stage_pct = stage_pct
        self.num_joints = num_joints

        self.drop_rate = tokenizer['encoder']['drop_rate']     
        self.enc_num_blocks = tokenizer['encoder']['num_blocks']
        self.enc_hidden_dim = tokenizer['encoder']['hidden_dim']
        self.enc_token_inter_dim = tokenizer['encoder']['token_inter_dim']
        self.enc_hidden_inter_dim = tokenizer['encoder']['hidden_inter_dim']
        self.enc_dropout = tokenizer['encoder']['dropout']

        self.dec_num_blocks = tokenizer['decoder']['num_blocks']
        self.dec_hidden_dim = tokenizer['decoder']['hidden_dim']
        self.dec_token_inter_dim = tokenizer['decoder']['token_inter_dim']
        self.dec_hidden_inter_dim = tokenizer['decoder']['hidden_inter_dim']
        self.dec_dropout = tokenizer['decoder']['dropout']

        self.token_num = tokenizer['codebook']['token_num']
        self.token_class_num = tokenizer['codebook']['token_class_num']
        self.token_dim = tokenizer['codebook']['token_dim']
        self.decay = tokenizer['codebook']['ema_decay']

        self.invisible_token = nn.Parameter(
            torch.zeros(1, 1, self.enc_hidden_dim))
        trunc_normal_(self.invisible_token, mean=0., std=0.02, a=-0.02, b=0.02)

        self.start_embed = nn.Linear(2, self.enc_hidden_dim)
        
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.num_joints, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        
        self.token_mlp = nn.Linear(self.num_joints, self.token_num)
        self.feature_embed = nn.Linear(self.enc_hidden_dim, self.token_dim)

        ## 
        self.decoder = CodebookDecoder(tokenizer)
        
        self.loss = build_loss(tokenizer['loss_keypoint'])

    def forward(self, joints, train=True):
        """Forward function. """

        # Encoder of Tokenizer, Get the PCT groundtruth class labels.
        joints_coord, joints_visible, bs \
            = joints[:,:,:-1], joints[:,:,-1].bool(), joints.shape[0]

        encode_feat = self.start_embed(joints_coord)

        if train and self.stage_pct == "tokenizer":
            rand_mask_ind = torch.rand(
                joints_visible.shape, device=joints_visible.device) > self.drop_rate
            joints_visible = torch.logical_and(rand_mask_ind, joints_visible) 

        mask_tokens = self.invisible_token.expand(bs, joints.shape[1], -1)
        w = joints_visible.unsqueeze(-1).type_as(mask_tokens)
        encode_feat = encode_feat * w + mask_tokens * (1 - w)

        # encoder
        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        
        encode_feat = encode_feat.transpose(2, 1)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
        encode_feat = self.feature_embed(encode_feat).flatten(0,1)
        
        # decoder & codebook
        part_token_feat, encodings, encoding_indices, recoverd_joints = \
            self.decoder(encode_feat, device=joints.device, bs=bs)
        # update codebook
        e_latent_loss = self.decoder.codebook_update(
            encodings, encode_feat, part_token_feat) if train else None


        return recoverd_joints, encoding_indices, e_latent_loss

    def get_loss(self, output_joints, joints, e_latent_loss):
        """Calculate loss for training tokenizer.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output_joints (torch.Tensor[NxKx3]): Recovered joints.
            joints(torch.Tensor[NxKx3]): Target joints.
            e_latent_loss(torch.Tensor[1]): Loss for training codebook.
        """

        losses = dict()

        kpt_loss, e_latent_loss = self.loss(output_joints, joints, e_latent_loss)

        losses['joint_loss'] = kpt_loss
        losses['e_latent_loss'] = e_latent_loss

        return losses

    def init_weights(self, pretrained=""):
        """Initialize model weights."""

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            assert (self.stage_pct == "classifier"), \
                "Training tokenizer does not need to load model"
            pretrained_state_dict = torch.load(pretrained, 
                            map_location=lambda storage, loc: storage)

            need_init_state_dict = {}

            for name, m in pretrained_state_dict['state_dict'].items():
                if 'keypoint_head.tokenizer.' in name:
                    name = name.replace('keypoint_head.tokenizer.', '')
                if name in parameters_names or name in buffers_names:
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=True)
        else:
            if self.stage_pct == "classifier":
                print('If you are training a classifier, '\
                    'must check that the well-trained tokenizer '\
                    'is located in the correct path.')
                
