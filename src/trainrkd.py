import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from os import path

from dataset import Yolo_dataset
from cfg import Cfg
from modelsv2_hint import Yolov4 as Yolov4_hint
from modelsv2 import Yolov4
from tool.darknet2yolo_hint import Darknet

from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator
from tool.distillation import DistillerRKD, DistillerIKD


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = (
            (bboxes_a[:, None, 0] + bboxes_a[:, None, 2])
            - (bboxes_b[:, 0] + bboxes_b[:, 2])
        ) ** 2 / 4 + (
            (bboxes_a[:, None, 1] + bboxes_a[:, None, 3])
            - (bboxes_b[:, 1] + bboxes_b[:, 3])
        ) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        # intersection bottom right
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        con_br = torch.max(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2
                )
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [
            [12, 16],
            [19, 36],
            [40, 28],
            [36, 75],
            [76, 55],
            [72, 146],
            [142, 110],
            [192, 243],
            [459, 401],
        ]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        (
            self.masked_anchors,
            self.ref_anchors,
            self.grid_x,
            self.grid_y,
            self.anchor_w,
            self.anchor_h,
        ) = ([], [], [], [], [], [])

        for i in range(3):
            all_anchors_grid = [
                (w / self.strides[i], h / self.strides[i]) for w, h in self.anchors
            ]
            masked_anchors = np.array(
                [all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32
            )
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]
            grid_x = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(batch, 3, fsize, 1)
                .to(device)
            )
            grid_y = (
                torch.arange(fsize, dtype=torch.float)
                .repeat(batch, 3, fsize, 1)
                .permute(0, 1, 3, 2)
                .to(device)
            )
            anchor_w = (
                torch.from_numpy(masked_anchors[:, 0])
                .repeat(batch, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )
            anchor_h = (
                torch.from_numpy(masked_anchors[:, 1])
                .repeat(batch, fsize, fsize, 1)
                .permute(0, 3, 1, 2)
                .to(device)
            )

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # target assignment
        tgt_mask = torch.zeros(
            batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes
        ).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(
            device=self.device
        )
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(
            self.device
        )
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(
            self.device
        )

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (
            self.strides[output_id] * 2
        )
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (
            self.strides[output_id] * 2
        )
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(
                truth_box.cpu(), self.ref_anchors[output_id], CIoU=True
            )

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])
            best_n_all = anchor_ious_all.argmax(dim=1)
            best_n = best_n_all % 3
            best_n_mask = (
                (best_n_all == self.anch_masks[output_id][0])
                | (best_n_all == self.anch_masks[output_id][1])
                | (best_n_all == self.anch_masks[output_id][2])
            )

            if sum(best_n_mask) == 0:
                continue

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = pred_best_iou > self.ignore_thre
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            obj_mask[b] = ~pred_best_iou

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(
                        torch.int16
                    ).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti]
                        / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0]
                        + 1e-16
                    )
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti]
                        / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1]
                        + 1e-16
                    )
                    target[b, a, j, i, 4] = 1
                    target[
                        b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()
                    ] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(
                        2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize
                    )
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0

        for output_id, output in enumerate(xin):
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            #print("{0}, {1}, {2}, {3}".format(batchsize, self.n_anchors, fsize, n_ch))

            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

            # logistic activation for xy, obj, cls
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
                output[..., np.r_[:2, 4:n_ch]]
            )

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[output_id]
            pred[..., 1] += self.grid_y[output_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(
                pred, labels, batchsize, fsize, n_ch, output_id
            )

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            output[..., 2:4] *= tgt_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            loss_xy += F.binary_cross_entropy(
                input=output[..., :2],
                target=target[..., :2],
                weight=tgt_scale * tgt_scale,
                reduction="sum",
            )
            loss_wh += (
                F.mse_loss(
                    input=output[..., 2:4], target=target[..., 2:4], reduction="sum"
                )
                / 2
            )
            loss_obj += F.binary_cross_entropy(
                input=output[..., 4], target=target[..., 4], reduction="sum"
            )
            loss_cls += F.binary_cross_entropy(
                input=output[..., 5:], target=target[..., 5:], reduction="sum"
            )
            loss_l2 += F.mse_loss(input=output, target=target, reduction="sum")

        loss = loss_xy + loss_wh + loss_obj + loss_cls
        #print('Loss ==========', loss)

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(teacher, 
    student,
    teacherweightfile,
    device,
    config,
    epochs=5,
    batch_size=1,
    save_cp=True,
    log_step=20,
    img_scale=0.5
):
    distilltype = config.distilltype # 'fkd', 'rkd', 'rskd', 'mkd'
    temperature = config.temperature
    alpha=config.alpha
    l = config.norm
    model = student
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch // config.subdivisions,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=val_collate,
    )

    writer = SummaryWriter(
        log_dir=config.TRAIN_TENSORBOARD_DIR,
        filename_suffix=f"OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}",
        comment=f"OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}",
    )

    global_step = 0
    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:
    """
    )
    # Get teacher model
    def get_teacher(weightfile):
        if config.TRAIN_OPTIMIZER.lower() == 'adam':
            optimizer = optim.Adam(
                teacher.parameters(),
                lr=config.learning_rate / config.batch,
                betas=(0.9, 0.999),
                eps=1e-08,
        )
        elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
            optimizer = optim.SGD(
                params=model.parameters(),
                lr=config.learning_rate / config.batch,
                momentum=config.momentum,
                weight_decay=config.decay,
            )
        if path.exists(weightfile):
            checkpoint = torch.load(weightfile, device)
            teacher.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        else:
            print('==================== No teacher model found ======================')
        return teacher

    # learning rate setup
    def burnin_schedule(i):
        if i < config.burn_in:
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]:
            factor = 1.0
        elif i < config.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(
        device=device,
        batch=config.batch // config.subdivisions,
        n_classes=config.classes,
    )

    save_prefix = "Yolov4_epoch"
    saved_models = deque()

    checkpoint_path = config.start_checkpoint_path
    if path.exists(checkpoint_path):
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"This is the current epoch {epoch}")
    else:
        epoch = 0

    model.train()
    teacher = get_teacher(teacherweightfile)
    for epoch in range(epoch, epochs):
        epoch_loss = 0
        epoch_step = 0
        bboxes_pred = []
        guidedlayer_fm = []

        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img", ncols=50
        ) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)
                y = teacher(images)
                x = model(images)
                

                teacher_predictions = y[0:3]
                hint_fm = y[3 : len(y)]
                bboxes_pred = x[0:3]  # Hint feature maps
                guidedlayer_fm = x[3 : len(x)]  # Guided feature maps
                distillation_loss = 0 
                hintlossrkd, hintlossfkd, hintloss = 0, 0, 0
                

                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(
                    bboxes_pred, bboxes
                )
                #---------------------------------------------------#
                #   response based distillation
                #---------------------------------------------------#
                if distilltype == 'rskd':
                    for j in range(len(bboxes_pred)):  
                        #print(bboxes_pred[j].shape, teacher_predictions[j].shape)
                        d = DistillerIKD(bboxes_pred[j], teacher_predictions[j] , temperature, l) 
                        # set distanceloss or/an angleloss in d to choose if relational mat 
                        distillation_loss += d.rskd_loss('s','t')

                    studentloss = loss
                    loss = alpha*loss + (1-alpha)*(distillation_loss)
                    print(f'Response based distiallation Loss: {distillation_loss} ======= Student Loss: {studentloss}====== General Loss {loss}')


                #---------------------------------------------------#
                #   feature based distillation
                #---------------------------------------------------#
                elif distilltype == 'fkd':
                    # Logits Loss
                    for j in range(len(bboxes_pred)):  
                        #print(bboxes_pred[j].shape, teacher_predictions[j].shape)
                        d = DistillerIKD(bboxes_pred[j], teacher_predictions[j] , temperature, l)
                        distillation_loss += d.rskd_loss('s','t')

                    # Hint Loss
                    for j in range(len(hint_fm)):  
                        #print(guidedlayer_fm[j].shape, hint_fm[j].shape)
                        d = DistillerIKD(guidedlayer_fm[j], hint_fm[j] , temperature, l)
                        hintloss += d.hint_loss('s','t')

                    # Student Loss
                    studentloss = loss

                    loss = config.beta * studentloss + config.gamma * distillation_loss + config.sigma * hintloss
                    print(
                        f"Feature based distiallation: ========= Logits Loss: {distillation_loss} ======= Student Loss: {studentloss}====== Hint Loss L{l}: {hintloss} ===== Overall Loss: {loss}"
                    )

                #---------------------------------------------------#
                #   relational distillation
                #---------------------------------------------------#
                elif distilltype == 'rkd':
                    for j in range(len(bboxes_pred)):  
                        #print(bboxes_pred[j].shape, teacher_predictions[j].shape)
                        d = DistillerIKD(bboxes_pred[j], teacher_predictions[j] , temperature, l)
                        distillation_loss += d.rskd_loss('s','t')
                    #print('###y, ####x ', len(y), len(x))
                    a = DistillerRKD(guidedfeaturemaps=guidedlayer_fm, 
                    hintfeaturemaps=hint_fm, angleloss=1, distanceloss=1)
                    hintloss = a.relationalloss()
                    studentloss = loss
                    loss = (
                        config.beta * studentloss
                        + config.gamma * distillation_loss
                        + config.sigma * hintloss
                    )
                    print(
                        f"Relation based distiallation Loss: == Logits Loss: {distillation_loss} ======= Student Loss: {studentloss}====== Hint Loss L{l}: {hintloss} ===== Overall Loss: {loss}"
                    )

                #---------------------------------------------------#
                #   multitype distillation
                #---------------------------------------------------#
                elif distilltype == 'mkd':
                    for j in range(len(bboxes_pred)):  
                        #print(bboxes_pred[j].shape, teacher_predictions[j].shape)
                        d = DistillerIKD(bboxes_pred[j], teacher_predictions[j] , temperature, l)
                        distillation_loss += d.rskd_loss('s','t')

                    a = DistillerRKD(guidedfeaturemaps=guidedlayer_fm, 
                    hintfeaturemaps=hint_fm, angleloss=1, distanceloss=1)
                    hintlossrkd = a.relationalloss()
                    
                    # Hint Loss
                    for j in range(len(hint_fm)):  
                        print(guidedlayer_fm[j].shape, hint_fm[j].shape)
                        d = DistillerIKD(guidedlayer_fm[j], hint_fm[j] , temperature, l)
                        hintlossfkd += d.hint_loss('s','t')
                    
                    studentloss = loss
                    loss = (
                        config.beta * studentloss
                        + config.gamma * distillation_loss
                        + config.sigma1 * hintlossrkd
                        + config.sigma2 * hintlossfkd
                    )
                    print(
                        f"multitype distiallation Loss: == Logits Loss: {distillation_loss} ======= Student Loss: {studentloss}====== Hint Loss L{l}: {hintloss} ===== Overall Loss: {loss}"
                    )

                loss.backward()

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar("train/Loss", loss.item(), global_step)
                    writer.add_scalar("train/loss_xy", loss_xy.item(), global_step)
                    writer.add_scalar("train/loss_wh", loss_wh.item(), global_step)
                    writer.add_scalar("train/loss_obj", loss_obj.item(), global_step)
                    writer.add_scalar("train/loss_cls", loss_cls.item(), global_step)
                    writer.add_scalar("train/loss_l2", loss_l2.item(), global_step)
                    writer.add_scalar(
                        "lr", scheduler.get_lr()[0] * config.batch, global_step
                    )
                    pbar.set_postfix(
                        **{
                            "loss (batch)": loss.item(),
                            "loss_xy": loss_xy.item(),
                            "loss_wh": loss_wh.item(),
                            "loss_obj": loss_obj.item(),
                            "loss_cls": loss_cls.item(),
                            "loss_l2": loss_l2.item(),
                            "lr": scheduler.get_lr()[0] * config.batch,
                        }
                    )
                    logging.debug(
                        "Train step_{}: loss : {},loss xy : {},loss wh : {},"
                        "loss obj : {}，loss cls : {},loss l2 : {},lr : {}".format(
                            global_step,
                            loss.item(),
                            loss_xy.item(),
                            loss_wh.item(),
                            loss_obj.item(),
                            loss_cls.item(),
                            loss_l2.item(),
                            scheduler.get_lr()[0] * config.batch,
                        )
                    )

                pbar.update(images.shape[0])

            eval_model = Yolov4(
                        wpath=config.mnv2_pretrained,
                        n_classes=cfg.classes,
                        inference=True
                            )

            if torch.cuda.device_count() > 1:
                eval_model.load_state_dict(model.module.state_dict())
                eval_model.load_state_dict(model.module.state_dict())
            else:
                eval_model.load_state_dict(model.state_dict(), strict=False)
            eval_model.to(device)
            evaluator = evaluate(eval_model, val_loader, config, device)
            del eval_model

            stats = evaluator.coco_eval["bbox"].stats
            writer.add_scalar("train/AP", stats[0], global_step)
            writer.add_scalar("train/AP50", stats[1], global_step)
            writer.add_scalar("train/AP75", stats[2], global_step)
            writer.add_scalar("train/AP_small", stats[3], global_step)
            writer.add_scalar("train/AP_medium", stats[4], global_step)
            writer.add_scalar("train/AP_large", stats[5], global_step)
            writer.add_scalar("train/AR1", stats[6], global_step)
            writer.add_scalar("train/AR10", stats[7], global_step)
            writer.add_scalar("train/AR100", stats[8], global_step)
            writer.add_scalar("train/AR_small", stats[9], global_step)
            writer.add_scalar("train/AR_medium", stats[10], global_step)
            writer.add_scalar("train/AR_large", stats[11], global_step)

            from datetime import datetime

            savetime = datetime.now()

            statstitle = [
                "=========== Results of epoch: ",
                "============ Date",
                "AP: ",
                "AP50: ",
                "AP75: ",
                "AP_small: ",
                "AP_medium: ",
                "AP_large: ",
                "AR1: ",
                "AR10: ",
                "AR100: ",
                "AR_small: ",
                "AR_medium: ",
                "AR_large: ",
            ]
            statsval = [
                epoch,
                savetime,
                stats[0],
                stats[1],
                stats[2],
                stats[3],
                stats[4],
                stats[5],
                stats[6],
                stats[7],
                stats[8],
                stats[9],
                stats[10],
                stats[11],
            ]


            if distilltype == 'mkd':
                cachefile = "trainresultmkd.txt"
            elif distilltype == 'rkd':
                cachefile = "trainresultrkd.txt"
            elif distilltype == 'rskd':
                cachefile = "trainresultrskd.txt"
            elif distilltype == 'fkd':
                cachefile = "trainresultmkd.txt"
            else:
                cachefile = "trainresult.txt"               
            with open(cachefile, "a") as f:
                for i in range(12):
                    f.write("\n")
                    f.write(statstitle[i] + str(statsval[i]))

            if save_cp:
                try:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info("Created checkpoint directory")
                except OSError:
                    pass
                save_path = os.path.join(
                    config.checkpoints, f"{save_prefix}{epoch + 1}.pth"
                )
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(
                        model.models,
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        save_path,
                    )
                else:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        save_path,
                    )
                logging.info(f"Checkpoint {epoch + 1} saved !")
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f"failed to remove {model_to_remove}")

    writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """finished, tested"""
    model.eval()

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt="coco")
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"], bbox_fmt="coco")

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)
        model_time = time.time() - model_time

        res = {}
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            test = boxes[..., 0]
            test1 = boxes[..., 1]
            boxes[..., 2:] = (
                boxes[..., 2:] - boxes[..., :2]
            )  # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            x = boxes[..., 2:]
            boxes[..., 0] = boxes[..., 0] * img_width
            boxes[..., 1] = boxes[..., 1] * img_height
            boxes[..., 2] = boxes[..., 2] * img_width
            boxes[..., 3] = boxes[..., 3] * img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(
        description="Train the Model on images and target masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.001,
        help="Learning rate",
        dest="learning_rate",
    )
    parser.add_argument(
        "-distilltype",
        type=str,
        default="rkd",
        help="knowledge distillation type (rkd, mkd, rskd, fkd)",
        dest="distilltype",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=None,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "-g", "--gpu", metavar="G", type=str, default="0", help="GPU", dest="gpu"
    )
    parser.add_argument(
        "-dir",
        "--data-dir",
        type=str,
        default=None,
        help="dataset dir",
        dest="dataset_dir",
    )
    parser.add_argument(
        "-pretrained", type=str, default=None, help="pretrained yolov4.conv.137"
    )
    parser.add_argument("-classes", type=int, default=80, help="dataset classes")
    parser.add_argument(
        "-optimizer",
        type=str,
        default="adam",
        help="training optimizer",
        dest="TRAIN_OPTIMIZER",
    )
    parser.add_argument(
        "-iou-type",
        type=str,
        default="iou",
        help="iou type (iou, giou, diou, ciou)",
        dest="iou_type",
    )
    parser.add_argument(
        "-keep-checkpoint-max",
        type=int,
        default=10,
        help="maximum number of checkpoints to keep. If set 0, all checkpoints will be kept",
        dest="keep_checkpoint_max",
    )

    parser.add_argument(
        "--scheduler_step_size", type=int, help="step size of the scheduler", default=15
    )
    args = vars(parser.parse_args())

    cfg.update(args)

    return edict(cfg)


def init_logger(
    log_file=None, log_dir=None, log_level=logging.INFO, mode="w", stdout=True
):

    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S")

    fmt = "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s"
    if log_dir is None:
        log_dir = "~/temp/log/"
    if log_file is None:
        log_file = "log_" + get_date_str() + ".txt"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    print("log file path:" + log_file)

    logging.basicConfig(
        level=logging.DEBUG, format=fmt, filename=log_file, filemode=mode
    )

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")


if __name__ == "__main__":
    logging = init_logger(log_dir="log")
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    teacher = Darknet(cfg.cfgfile)

    if cfg.distilltype=='mkd':
        print("*********************************************************************")
        print("********************    Distillation using multi-type KD ************")
        print("******************** Teacher: Darknet, Student: MobilenetV2 *********")
        print("*********************************************************************")        
    elif cfg.distilltype=='fkd':
        print("*********************************************************************")
        print("********************  Distillation using feature based KD ***********")
        print("******************** Teacher: Darknet, Student: MobilenetV2 *********")
        print("*********************************************************************")      
    elif cfg.distilltype=='rskd':
        print("*********************************************************************")
        print("********************  Distillation using response based KD **********")
        print("******************** Teacher: Darknet, Student: MobilenetV2 *********")
        print("*********************************************************************")      
    elif cfg.distilltype=='rkd':
        print("*********************************************************************")
        print("********************  Distillation using relation based KD **********")
        print("******************** Teacher: Darknet, Student: MobilenetV2 *********")      
        print("*********************************************************************")      
    else:
        print("*********************************************************************")
        print("********************      No Distillation      **********************")
        print("******************** Train YOLO-V4 MobilenetV2 **********************")      
        print("*********************************************************************")      

    model_train = Yolov4_hint(
                wpath=cfg.mnv2_pretrained,
                n_classes=cfg.classes
            )
    
    if torch.cuda.device_count() > 1:
        model_train = torch.nn.DataParallel(model_train)
    model_train.to(device=device)
    teacher.to(device=device)

    try:
        train(
            teacher = teacher,
            teacherweightfile = cfg.teacherweightfile,
            student=model_train,
            config=cfg,
            epochs=cfg.TRAIN_EPOCHS,
            device=device,
        )
    except KeyboardInterrupt:
        if isinstance(model_train, torch.nn.DataParallel):
            torch.save(model_train.module.state_dict(), "INTERRUPTED.pth")
        else:
            torch.save(model_train.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
