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
from mmpose.models import builder
from mmpose.models.pose_estimators.base import BasePoseEstimator
from mmengine.model.base_model import BaseModel
from mmpose.models.data_preprocessors import PoseDataPreprocessor

from .pct_tokenizer import Tokenizer ## PCT tokenizer for STAGE I
from models import build_backbone, build_head

from mmengine.structures import InstanceData, PixelData
import os

@MODELS.register_module()
class PCT(BaseModel):  ## BasePoseEstimator 대신 BaseModel 상속
    def __init__(self,
                  backbone,
                  keypoint_head=None,
                  test_cfg=None,
                  data_preprocessor=None,
                  init_cfg=None,
                  pretrained=None):
        super().__init__()
        
        self.test_cfg = test_cfg if test_cfg else {}

        self.stage_pct = keypoint_head['stage_pct']
        assert self.stage_pct in ["tokenizer", "classifier"]
       
        self.tokenizer = Tokenizer(stage_pct=self.stage_pct, tokenizer=keypoint_head['tokenizer'])
        self.tokenizer.init_weights(pretrained=keypoint_head['tokenizer']['ckpt'])

        # self.tokenizer.init_weights(pretrained="")

        if self.stage_pct == "tokenizer":
            # For training tokenizer
            keypoint_head['loss_keypoint'] \
                = keypoint_head['tokenizer']['loss_keypoint']
                
        if self.stage_pct == "classifier":
            # For training classifier
            self.keypoint_head = build_head(keypoint_head)
            self.keypoint_head.init_weights()
            self.keypoint_head.tokenizer.init_weights(pretrained=keypoint_head['tokenizer']['ckpt'])
            
            # backbone is only needed for training classifier
            self.backbone = build_backbone(backbone)
            self.backbone.init_weights(pretrained)

        self.flip_test = test_cfg.get('flip_test', True)
        self.dataset_name = test_cfg.get('dataset_name', 'AP10K')

    def forward(self, inputs, data_samples, mode: str = 'tensor'): # -> ForwardResults:
        # print(f'Input Image test: {type(inputs), len(inputs)}')
        # print(f'Input element -> Input[0].shape == {inputs[0].shape}')
           
        DEVICE = next(self.parameters()).device
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0).to(DEVICE)  # Stack inputs along the batch dimension

        if inputs.dtype != torch.float32:
            inputs = inputs.float()  # Ensure inputs are float
        # print(f'Stacked Input shape: {type(inputs)}, {inputs.shape}')
        img_metas, joints_3d, joints_3d_visible = [], [], []

        for sample in data_samples:
            img_metas.append(sample.metainfo)
            joints_3d.append(torch.tensor(sample.gt_instances.transformed_keypoints))
            joints_3d_visible.append(torch.tensor(sample.gt_instances.keypoints_visible))

        joints_3d = torch.cat(joints_3d, dim=0).to(DEVICE)
        joints_3d_visible = torch.cat(joints_3d_visible, dim=0).to(DEVICE)

        joints_3d_visible = joints_3d_visible.unsqueeze(-1)
        joints = torch.cat((joints_3d, joints_3d_visible), dim=-1)

        # print(f"joints_3d shape: {joints_3d.shape}")
        # print(f"joints_3d_visible shape: {joints_3d_visible.shape}")
        # print(f"joints shape: {joints.shape}")

        if mode == 'loss':
            return self.forward_train(inputs, joints, img_metas)
        elif mode == 'predict':
            return self.forward_test(inputs, joints, img_metas, data_samples)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_train(self, inputs, joints, img_metas):
        """Defines the computation performed at every call when training."""
        # print(f"inputs shape: {type(inputs)}, {inputs.shape}")
        output = None if self.stage_pct == "tokenizer" else self.backbone(inputs)
        # print(f"output len: {type(output)}, {len(output)}")
        # print(f"output[0] shape: {output[0].shape}")
        # print(f"joints shape: {joints.shape}")
        extra_output = None # if not self.image_guide else self.extra_backbone(img)
        p_logits, p_joints, g_logits, e_latent_loss = self.keypoint_head(output, extra_output, joints)

        losses = dict()

        if self.stage_pct == "classifier":
            keypoint_losses = self.keypoint_head.get_loss(p_logits, p_joints, g_logits, joints)
            losses.update(keypoint_losses)
            p_logits = p_logits.view(-1, p_logits.size(-1))  # [N*M, V]로 변환
            g_logits = g_logits.view(-1)  # [N*M]으로 변환

            topk = (1,2,5)
            keypoint_accuracy = self.get_class_accuracy(p_logits, g_logits, topk)
            kpt_accs = {}
            for i in range(len(topk)):
                kpt_accs['top%s-acc' % str(topk[i])] \
                    = keypoint_accuracy[i]
            losses.update(kpt_accs)            

        elif self.stage_pct == "tokenizer":
            keypoint_losses = self.tokenizer.get_loss(p_joints, joints, e_latent_loss)
            losses.update(keypoint_losses)
        return losses

    def get_class_accuracy(self, output, target, topk):
        
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [
            correct[:k].reshape(-1).float().sum(0) \
                * 100. / batch_size for k in topk]

    def forward_test(self, img, joints, img_metas, data_samples): # -> SampleList
        """Defines the computation performed at every call when testing."""
        assert img.size(0) == len(img_metas)  # Ensure img size matches img_metas length
        results = {}
        # print(f"img shape: {type(img)}, {img.shape}")

        batch_size, _, img_height, img_width = img.shape
        if batch_size > 1:
            assert 'id' in img_metas[0]

        output = None if self.stage_pct == "tokenizer" else self.backbone(img) 
        extra_output = None
        # extra_output = self.extra_backbone(img) \
        #     if self.image_guide and self.stage_pct == "tokenizer" else None
        # print(f'joints shape: {joints.shape}')
        p_joints, encoding_scores = self.keypoint_head(output, extra_output, joints, train=False)
        score_pose = joints[:,:,2:] if self.stage_pct == "tokenizer" else \
            encoding_scores.mean(1, keepdim=True).repeat(1,p_joints.shape[1],1)

        # if self.flip_test:
        #     FLIP_INDEX = {'COCO': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15], \
        #             'CROWDPOSE': [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13], \
        #             'OCCLUSIONPERSON':[0, 4, 5, 6, 1, 2, 3, 7, 8, 12, 13, 14, 9, 10, 11],\
        #             'MPII': [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10],\
        #             'AP10K':[1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]}

        #     img_flipped = img.flip(3)

        #     features_flipped = None if self.stage_pct == "tokenizer" \
        #         else self.backbone(img_flipped) 
        #     extra_output_flipped = None

        #     if joints is not None:
        #         joints_flipped = joints.clone()
        #         joints_flipped = joints_flipped[:,FLIP_INDEX[self.dataset_name],:]
        #         joints_flipped[:,:,0] = img.shape[-1] - 1 - joints_flipped[:,:,0]
        #     else:
        #         joints_flipped = None
                
        #     p_joints_f, encoding_scores_f = \
        #         self.keypoint_head(features_flipped, \
        #             extra_output_flipped, joints_flipped, train=False)

        #     p_joints_f = p_joints_f[:,FLIP_INDEX[self.dataset_name],:]
        #     p_joints_f[:,:,0] = img.shape[-1] - 1 - p_joints_f[:,:,0]

        #     score_pose_f = joints[:,:,2:] if self.stage_pct == "tokenizer" else \
        #         encoding_scores_f.mean(1, keepdim=True).repeat(1,p_joints.shape[1],1)

        #     p_joints = (p_joints + p_joints_f)/2.0
        #     score_pose = (score_pose + score_pose_f)/2.0

        batch_size = len(img_metas)

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

        # p_joints = p_joints.cpu().numpy()
        # score_pose = score_pose.cpu().numpy()
        # for i in range(p_joints.shape[0]):
        #     p_joints[i] = transform_preds(
        #         p_joints[i], c[i], s[i], [img.shape[-1], img.shape[-2]], use_udp=False)
        
        # all_preds = np.zeros((batch_size, p_joints.shape[1], 3), dtype=np.float32)
        # all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        # all_preds[:, :, 0:2] = p_joints
        # all_preds[:, :, 2:3] = score_pose
        # all_boxes[:, 0:2] = c[:, 0:2]
        # all_boxes[:, 2:4] = s[:, 0:2]
        # all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        # all_boxes[:, 5] = score

        # # final_preds = {}
        # # final_preds['preds'] = all_preds
        # # final_preds['boxes'] = all_boxes
        # # final_preds['image_paths'] = image_paths
        # # final_preds['bbox_ids'] = bbox_ids
        # # results.update(final_preds)
        # # results['output_heatmap'] = None

        # # return results
        # for i in range(batch_size):
        #     pred_instances = InstanceData()
        #     pred_instances.keypoints = all_preds[i, :, 0:2]
        #     pred_instances.keypoint_scores = all_preds[i, :, 2]
        #     pred_instances.bboxes = all_boxes[i, :4]
        #     pred_instances.bbox_scores = all_boxes[i, 5]

        #     data_samples[i].pred_instances = pred_instances

        # return data_samples    
        
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
            # expand instance dimension (17, 2) => (1, 17, 2)
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


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.

    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = coords.copy()
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords