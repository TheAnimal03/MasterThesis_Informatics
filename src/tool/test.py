import torch
import sys
import cv2
from torch import nn
import torch.nn.functional as F
import models as model
from  cfg import Cfg
import argparse
from easydict import EasyDict as edict


batchsize = Cfg.batch
anchors = 3
fsize = Cfg.width
n_ch = 3

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')
    #parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="train label path")
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)

    return edict(cfg)
   
# cfg = get_args(**Cfg)
# input = torch.rand(batchsize, anchors, fsize, fsize, n_ch)
# model = model.Yolov4(cfg.pretrained, n_classes=cfg.classes)
# print(model)

namesfile = None
if len(sys.argv) == 6:
    n_classes = int(sys.argv[1])
    weightfile = sys.argv[2]
    imgfile = sys.argv[3]
    height = int(sys.argv[4])
    width = int(sys.argv[5])
elif len(sys.argv) == 7:
    n_classes = int(sys.argv[1])
    weightfile = sys.argv[2]
    imgfile = sys.argv[3]
    height = int(sys.argv[4])
    width = int(sys.argv[5])
    namesfile = sys.argv[6]
else:
    print('Usage: ')
    print('  python models.py num_classes weightfile imgfile namefile')

model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
model.load_state_dict(pretrained_dict)

use_cuda = True
if use_cuda:
    model.cuda()

img = cv2.imread(imgfile)

# Inference input size is 416*416 does not mean training size is the same
# Training size could be 608*608 or even other sizes
# Optional inference sizes:
#   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
#   Width in {320, 416, 512, 608, ... 320 + 96 * m}
sized = cv2.resize(img, (width, height))
sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect

for i in range(2):  # This 'for' loop is for speed check
                    # Because the first iteration is usually longer
    boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

if namesfile == None:
    if n_classes == 20:
        namesfile = 'data/voc.names'
    elif n_classes == 80:
        namesfile = 'data/coco.names'
    else:
        print("please give namefile")

class_names = load_class_names(namesfile)
plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)

