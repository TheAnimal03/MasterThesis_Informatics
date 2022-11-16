# Master Thesis: Knowlege Distillation

![](https://img.shields.io/static/v1?label=python&message=3.6|3.7&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=1.4&color=<COLOR>)

A PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code yolo PyTorch: https://github.com/AlexeyAB/darknet, https://github.com/Tianxiaomo/pytorch-YOLOv4
- More details: http://pjreddie.com/darknet/yolo/, https://pytorch.org/hub/pytorch_vision_mobilenet_v2/

A PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code yolo PyTorch: https://github.com/AlexeyAB/darknet, https://github.com/Tianxiaomo/pytorch-YOLOv4
- More details: http://pjreddie.com/darknet/yolo/

# 0. Design Understanding
## Folders
```
├── Literatur           collection for OD and KD 
├── Vorbereitung        proposal for thesis
├── source (src)        
|   ├── pytorch-YOLOv4  main folder for the implemantation
```
## Code inside pytorch-YOLOv4 
```
├── README.md
├── dataset.py            dataset
├── demo.py               demo to run pytorch --> tool/darknet2pytorch
├── demo_darknet2onnx.py  tool to convert into onnx --> tool/darknet2pytorch
├── demo_pytorch2onnx.py  tool to convert into onnx
├── models.py             model for pytorch
├── train.py              train models.py (Backbone Darknet)
├── trainmnv2.py          train models.py (Backbone mobilenetv2)
├── cfg.py                cfg.py for train
├── cfg                   cfg --> darknet2pytorch, mobilenetv2
├── data                  --> COCO 2017
├── weight                --> darknet2pytorch
├── tool
│   ├── camera.py           a demo camera
│   ├── coco_annotation.py       coco dataset generator for annotations
│   ├── config.py
│   ├── darknet2pytorch.py
├── ├── mobilenetv2.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```

# 1. Data Preprocessing
## 1.1 Download coco 2017
    - coco train data (http://images.cocodataset.org/zips/train2017.zip)
    - coco validation data (http://images.cocodataset.org/zips/val2017.zip)
## 1.2 Transform data
    For coco dataset, you can use tool/coco_annotation.py.
    ```
    # train.txt
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ...
    ```
# 2. Train
## 2.1 Set train parameters and link for train annotations file.
    you can set parameters in:
    - cfg.py for training with Darknet.
    - cfgmnv2.py for training with MobileNetv2.
    
## 2.2 Start training
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```
