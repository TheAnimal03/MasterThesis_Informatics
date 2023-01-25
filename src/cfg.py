import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
Cfg.anchors = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

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
Cfg.TRAIN_EPOCHS = 320

# All Links
Cfg.teacherweightfile =r'/content/gdrive/MyDrive/Uni/MA/pytorch-YOLOv4/IntermediateResults/trainresults/Yolov4_epoch206.pth' 
Cfg.train_label = r'/content/gdrive/MyDrive/Uni/MA/coco/train2017.txt' #os.path.join(_BASE_DIR, 'data', 'train.txt')
Cfg.val_label = r'/content/gdrive/MyDrive/Uni/MA/coco/val2017.txt' #os.path.join(_BASE_DIR, 'data' ,'val.txt')
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.DATA_PATH = r'/content/gdrive/MyDrive/Uni/MA/coco/'
Cfg.start_checkpoint_path = r"/content/gdrive/MyDrive/Uni/MA/pytorch-YOLOv4/checkpoints/Yolov4_epoch360.pth"
Cfg.mnv2_pretrained = r"/content/gdrive/MyDrive/Uni/MA/pytorch-YOLOv4/cfg/mobilenetv2-c5e733a8.pth"

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

# Distiallation parameters
Cfg.distilltype = 'rkd' # 'fkd', 'rkd', 'rskd', 'mkd'
Cfg.norm = 2 # 1 or 2
Cfg.temperature = 0.1
Cfg.alpha = 0.1
Cfg.beta = 0.05
Cfg.sigma = 0.3
Cfg.sigma1 = 0.1
Cfg.sigma2 = 0.2
Cfg.gamma = 0.05
Cfg.norm = 2 # 1 or 2
