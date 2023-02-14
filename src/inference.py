from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
from modelsv2 import Yolov4
import torch
import argparse
from cfg import Cfg as cfg
from torch.hub import load_state_dict_from_url
import numpy as np

"""hyper parameters"""
use_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model_url = ("/content/gdrive/MyDrive/Uni/MA/pytorch-YOLOv4/cfg/mobilenetv2-c5e733a8.pth")

def detect_cv2(cfgfile, weightfile, imgfile, out=True):
    import cv2
    model_url = ("/content/gdrive/MyDrive/Uni/MA/pytorch-YOLOv4/cfg/mobilenetv2-c5e733a8.pth")
    d = Darknet(cfgfile
            )
    m = Yolov4(
            wpath=model_url,
            n_classes=cfg.classes
        )

    if use_cuda:
        m.cuda()
        print(f'device: CUDA')

    num_classes = cfg.classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    
    sized = cv2.resize(img, (cfg.width, cfg.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    print(sized.shape)
    images = 1000
    k = 0
    inferdk =[]
    infermn =[]

    while k < images:
      infer = []
      # Mobile
      start = time.time()
      infm = do_simpledetect(m, sized, 0.4, 0.6, use_cuda)
      finish = time.time()

      #Mobile
      start = time.time()
      infd = do_simpledetect(d, sized, 0.4, 0.6, use_cuda)
      finish = time.time()

      infermn.append(infm)
      inferdk.append(infd)

      k+=1
    print (infermn, inferdk)  
    np.savetxt('data.csv', (infermn, inferdk), delimiter=';') 

    # if out == True:
    #     return boxes[0]
    # else:
    #     print('boxes[0]==============================', boxes[0])
    #     plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        
                        default=r'/content/gdrive/MyDrive/Uni/MA/pytorch-YOLOv4/checkpoints/Yolov4_epoch174.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/000000289343.jpg',
                        help='path of your image file.', dest='imgfile')
    # parser.add_argument('-torch', type=bool, default=false,
    #                     help='use torch weights')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
    #detect_cv2(args.weightfile, args.imgfile)

