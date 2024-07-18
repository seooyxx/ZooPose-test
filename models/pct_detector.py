import time
import torch
import numpy as np

import mmcv

from itertools import zip_longest
from typing import Optional

# from mmengine.registry import MODELS
from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)

from mmpose.models.pose_estimators.base import BasePoseEstimator
from mmengine.model.base_model import BaseModel
from mmpose.models.data_preprocessors import PoseDataPreprocessor

from .pct_tokenizer import Tokenizer ## PCT tokenizer for STAGE I

from mmengine.structures import InstanceData, PixelData


@MODELS.register_module()
class PCT(BaseModel):  ## BasePoseEstimator 대신 BaseModel 상속
    def __init__(self,
                  backbone,
                  neck=None,
                  head=None,
                  train_cfg=None,
                  test_cfg=None,
                  data_preprocessor=None,
                  init_cfg=None,
                  metainfo=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        self.stage_pct = head['stage_pct']
        assert self.stage_pct in ["tokenizer", "classifier"]
       
        self.tokenizer = Tokenizer(stage_pct=self.stage_pct, tokenizer=head['tokenizer'])
        self.tokenizer.init_weights(pretrained="")

        self.flip_test = test_cfg.get('flip_test', True)
        self.dataset_name = test_cfg.get('dataset_name', 'AP10K')


    def forward(self, inputs, data_samples, mode: str = 'tensor'): # -> ForwardResults:
           
        DEVICE = next(self.parameters()).device

        img_metas, joints_3d, joints_3d_visible = [], [], []


        ## 기존 PCT 데이터 형식으로 처리하기 위한 코드이며, (NEED : img_metas, joints_3d, joints_visible)
        ## 수정 시 그대로 data_samples에서 필요한 데이터들 꺼내 써도 될 것 같음
        ## 카테고리 정보 등은 metainfo에 있음

        for sample in data_samples:
            img_metas.append(sample.metainfo)
            joints_3d.append(torch.tensor(sample.gt_instances.transformed_keypoints))
            joints_3d_visible.append(torch.tensor(sample.gt_instances.keypoints_visible))

        joints_3d = torch.cat(joints_3d, dim=0).to(DEVICE)
        joints_3d_visible = torch.cat(joints_3d_visible, dim=0).to(DEVICE)

        joints_3d_visible = joints_3d_visible.unsqueeze(-1)
        joints = torch.cat((joints_3d, joints_3d_visible), dim=-1)

        if mode == 'loss':
            return self.forward_train(joints, img_metas)
        elif mode == 'predict':
            return self.forward_test(inputs, joints, img_metas, data_samples)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_train(self, joints, img_metas):
        """Defines the computation performed at every call when training."""

        # tokenizer
        output_joints, cls_label, e_latent_loss = self.tokenizer(joints, train=True)

        losses = dict()

        keypoint_losses = self.tokenizer.get_loss(output_joints, joints, e_latent_loss)
        losses.update(keypoint_losses)
        return losses

    # def get_class_accuracy(self, output, target, topk):
        
    #     maxk = max(topk)
    #     batch_size = target.size(0)
    #     _, pred = output.topk(maxk, 1, True, True)
    #     pred = pred.t()
    #     correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    #     return [
    #         correct[:k].reshape(-1).float().sum(0) \
    #             * 100. / batch_size for k in topk]

    def forward_test(self, img, joints, img_metas, data_samples): # -> SampleList
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas) # (batch_size, 3, 256, 256)

        results = {}

        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'id' in img_metas[0]
            
        # output = None if self.stage_pct == "tokenizer" \
        #     else self.backbone(img) 
        # extra_output = self.extra_backbone(img) \
        #     if self.image_guide and self.stage_pct == "tokenizer" else None
        
        p_joints, _, _ = self.tokenizer(joints, train=False)
        score_pose = joints[:,:,2:] #if self.stage_pct == "tokenizer" else encoding_scores.mean(1, keepdim=True).repeat(1,p_joints.shape[1],1)

        if self.flip_test:
            FLIP_INDEX = {'COCO': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], \
                    'CROWDPOSE': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13], \
                    'OCCLUSIONPERSON':[0, 4, 5, 6, 1, 2, 3, 7, 8, 12, 13, 14, 9, 10, 11],\
                    'MPII': [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10],\
                    'AP10K':[1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]}

            # img_flipped = img.flip(3)
    
            # # features_flipped = None if self.stage_pct == "tokenizer" \
            # #     else self.backbone(img_flipped) 
            # # extra_output_flipped = self.extra_backbone(img_flipped) \
            # #     if self.image_guide and self.stage_pct == "tokenizer" else None

            # features_flipped, extra_output_flipped = None, None

            if joints is not None:
                joints_flipped = joints.clone()
                joints_flipped = joints_flipped[:,FLIP_INDEX[self.dataset_name],:]
                joints_flipped[:,:,0] = img.shape[-1] - 1 - joints_flipped[:,:,0]
            else:
                joints_flipped = None

            p_joints_f, _, _ = self.tokenizer(joints, train=False)
                
            # p_joints_f, encoding_scores_f = \
            #     self.head(features_flipped, \
            #         extra_output_flipped, joints_flipped, train=False)

            p_joints_f = p_joints_f[:,FLIP_INDEX[self.dataset_name],:]
            p_joints_f[:,:,0] = img.shape[-1] - 1 - p_joints_f[:,:,0]

            score_pose_f = joints[:,:,2:] # if self.stage_pct == "tokenizer" else encoding_scores_f.mean(1, keepdim=True).repeat(1,p_joints.shape[1],1)

            p_joints = (p_joints + p_joints_f)/2.0
            score_pose = (score_pose + score_pose_f)/2.0

        batch_size = len(img_metas)

        # if 'bbox_id' in img_metas[0]:
        bbox_ids = [] if 'id' in img_metas[0] else None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)

        for i in range(batch_size):
            c[i, :] = img_metas[i]['input_center']
            s[i, :] = img_metas[i]['input_scale']
            image_paths.append(img_metas[i]['img_path'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['id'])

        recovered_joints = p_joints.detach().cpu().numpy()
        # score_pose = score_pose.detach().cpu().numpy()

        ## evaluation을 위해 data_samples에 prediction 추가
        if isinstance(recovered_joints, tuple):
            batch_pred_instances, batch_pred_fields = recovered_joints
        else:
            batch_pred_instances = recovered_joints
            batch_pred_fields = None

        results = self.add_pred_to_datasample(batch_pred_instances, batch_pred_fields, data_samples)

        return results

    def show_result(self):
        # Not implemented
        return None


    def add_pred_to_datasample(self, batch_pred_instances: InstanceList,
                               batch_pred_fields: Optional[PixelDataList],
                               batch_data_samples: SampleList) -> SampleList:
        """Add predictions into data samples.

        Args:
            batch_pred_instances (List[InstanceData]): The predicted instances
                of the input data batch
            batch_pred_fields (List[PixelData], optional): The predicted
                fields (e.g. heatmaps) of the input batch
            batch_data_samples (List[PoseDataSample]): The input data batch

        Returns:
            List[PoseDataSample]: A list of data samples where the predictions
            are stored in the ``pred_instances`` field of each data sample.
        """

        # 임시 스코어
        sc = np.ones((17, ))


        # print(len(batch_pred_instances), len(batch_data_samples))
        assert len(batch_pred_instances) == len(batch_data_samples)
        
        if batch_pred_fields is None:
            batch_pred_fields = []
        output_keypoint_indices = self.test_cfg.get('output_keypoint_indices',
                                                    None)

        #for pred_instances, pred_fields, data_sample in zip_longest(
        for prediction, pred_fields, data_sample in zip_longest(
                batch_pred_instances, batch_pred_fields, batch_data_samples):

            gt_instances = data_sample.gt_instances

            # convert keypoint coordinates from input space to image space
            input_center = data_sample.metainfo['input_center']
            input_scale = data_sample.metainfo['input_scale']
            input_size = data_sample.metainfo['input_size']

            ##
            pred_instances = InstanceData()
            # exapand instance dimension (17, 2) => (1, 17, 2)
            pred_instances.set_field(np.expand_dims(prediction, axis=0), "keypoints")
            pred_instances.set_field(np.expand_dims(sc, axis=0), "keypoint_scores")

            pred_instances.keypoints[..., :2] = \
                pred_instances.keypoints[..., :2] / input_size * input_scale \
                + input_center - 0.5 * input_scale
            
            if 'keypoints_visible' not in pred_instances:
                pred_instances.keypoints_visible = \
                    pred_instances.keypoint_scores

            if output_keypoint_indices is not None:
                # select output keypoints with given indices
                num_keypoints = pred_instances.keypoints.shape[1]
                for key, value in pred_instances.all_items():
                    if key.startswith('keypoint'):
                        pred_instances.set_field(
                            value[:, output_keypoint_indices], key)


            # add bbox information into pred_instances
            pred_instances.bboxes = gt_instances.bboxes
            pred_instances.bbox_scores = gt_instances.bbox_scores

            data_sample.pred_instances = pred_instances

            if pred_fields is not None:
                if output_keypoint_indices is not None:
                    # select output heatmap channels with keypoint indices
                    # when the number of heatmap channel matches num_keypoints
                    for key, value in pred_fields.all_items():
                        if value.shape[0] != num_keypoints:
                            continue
                        pred_fields.set_field(value[output_keypoint_indices], key)
                data_sample.pred_fields = pred_fields

        return batch_data_samples
    