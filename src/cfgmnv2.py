# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:05
@Author        : Tianxiaomo
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
from easydict import EasyDict
#from pathlib import Pathfrom pathlib import Path
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#src = Path.cwd()
#os.path.split(src, os.path.splitext(os.path.basename(src))[0])
#print(src)
#sfle, rfle = os.path.split(src)

#currdir = Path.cwd()

Cfg = EasyDict()

Cfg.use_darknet_cfg = True
Cfg.use_mobilenetv2_cfg = True
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')

Cfg.batch = 64
Cfg.subdivisions = 16
Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.cutmix = 0
Cfg.mosaic = 1

Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 80
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # box num
Cfg.TRAIN_EPOCHS = 1000
Cfg.train_label = '/content/gdrive/MyDrive/Uni/MA/coco/train2017_.txt' #r'Google Drive/My Drive/Uni/MA/coco/train2017_Lab.txt' #os.path.join(_BASE_DIR, 'coco', 'train2017_.txt') #r'G:\My Drive\Uni\MA\coco\train2017_.txt' #os.path.join(_BASE_DIR, 'data', 'train.txt')
Cfg.val_label = '/content/gdrive/MyDrive/Uni/MA/coco/val2017_.txt' #os.path.join(_BASE_DIR, 'data' ,'val.txt') #os.path.join(_BASE_DIR, 'coco','val2017_.txt') #r'G:\My Drive\Uni\MA\coco\val2017_.txt' #os.path.join(_BASE_DIR, 'data' ,'val.txt')
Cfg.TRAIN_OPTIMIZER = 'adam'
'''
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
'''

if Cfg.mosaic and Cfg.cutmix:
    Cfg.mixup = 4
elif Cfg.cutmix:
    Cfg.mixup = 2
elif Cfg.mosaic:
    Cfg.mixup = 3

Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg.keep_checkpoint_max = 10
