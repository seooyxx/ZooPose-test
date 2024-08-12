# ZooPose

## Environment
The code is developed using python 3.8 on Ubuntu 16.04. The code is developed and tested using 8 NVIDIA V100 GPU cards. Other platforms are not fully tested.

## Usage
### Installation
1. Clone this repo.
2. Setup conda environment:
   ```
   conda create -n PCT python=3.8 -y
   conda activate PCT
   pip install -r requirements.txt
   ```

### Data Preparation

To obtain the COCO dataset, it can be downloaded from the [COCO download](http://cocodataset.org/#download), and specifically the 2017 train/val files are required. Additionally, the person detection results can be acquired from the [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) repository. The resulting data directory should look like this:

### PCT

#### Stage I: Training Tokenizer

```
./tools/dist_train.sh configs/pct_[base/large/huge]_tokenizer.py 8
```
Aftering training tokenizer, you should move the well-trained tokenizer from the `work_dirs/pct_[base/large/huge]_tokenizer/epoch_50.pth` to the `weights/tokenizer/swin_[base/large/huge].pth` and then proceed to the next stage. Alternatively, you can change the config of classifier using `--cfg-options model.keypoint_head.tokenizer.ckpt=work_dirs/pct_[base/large/huge]_tokenizer/epoch_50.pth` to train the classifier.

#### Stage II: Training Classifier

```
./tools/dist_train.sh configs/pct_[base/large/huge]_classifier.py 8
```

Finally, you can test your model using the script below.
```
./tools/dist_test.sh configs/pct_[base/large/huge]_classifier.py work_dirs/pct_[base/large/huge]_classifier/epoch_210.pth 8 --cfg-options data.test.data_cfg.use_gt_bbox=False
```

#### Remove image guidance
Additionally, you can choose a cleaner PCT that removes image guidance. The benefit of this approach is that it doesn't require features from a backbone trained on COCO with heatmap supervision. Instead, it directly converts joint coordinates into compositional tokens, making it easier to perform various visualization and analysis tasks. This approach has a slightly reduced performance impact.
```
./tools/dist_train.sh configs/pct_base_woimgguide_tokenizer.py 8
./tools/dist_train.sh configs/pct_base_woimgguide_classifier.py 8
```

#### Demo

You need to install mmdet==2.26.0 and mmcv-full==1.7.0, and then use the following command to generate some image demos.
```
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH python vis_tools/demo_img_with_mmdet.py vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth configs/pct_[base/large/huge]_classifier.py weights/pct/swin_[base/large/huge].pth --img-root images/ --img your_image.jpg --out-img-root images/ --thickness 2
```
