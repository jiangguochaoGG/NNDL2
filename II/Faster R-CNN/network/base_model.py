import os
from abc import abstractmethod

import cv2, copy
import torch
import warnings
import sys
import ipdb
from utils.eval_metrics.eval_map import eval_detection_voc

from misc_utils import color_print, progress_bar, save_json
from options import opt, config
import misc_utils as utils
import numpy as np
from mscv import load_checkpoint, save_checkpoint
from mscv.image import tensor2im
from mscv.summary import write_loss, write_image, write_meters_loss
from utils.vis import visualize_boxes, visualize_boxes_pb

class BaseModel(torch.nn.Module):
    def __init__(self, config, kwargs):
        super(BaseModel, self).__init__()
        self.config = config
        for key, value in kwargs.items():
            setattr(self, key, value)


    def forward(self, sample, *args):
        if self.training:
            return self.update(sample, *args)
        else:
            image = sample
            return self.forward_test(image, *args)


    @abstractmethod
    def update(self, sample: dict, *args, **kwargs):
        """
        这个函数会计算loss并且通过optimizer.step()更新网络权重。
        """
        pass

    @abstractmethod
    def forward_test(self, image, *args):
        """
        这个函数会由输入图像给出一个batch的预测结果。

        Args:
            image: [b, 3, h, w] Tensor

        Returns:
            tuple: (batch_bboxes, batch_labels, batch_scores)

            batch_bboxes: [ [Ni, 4] * batch_size ]
                一个batch的预测框，xyxy格式
                batch_bboxes[i]，i ∈ [0, batch_size-1]

            batch_labels: [[N1], [N2] ,... [N_bs]]
                一个batch的预测标签，np.int32格式

            batch_scores: [[N1], [N2] ,... [N_bs]]
                一个batch的预测分数，np.float格式
        """
        pass

    def eval_mAP(self, dataloader, epoch, writer, logger, data_name='val'):
        # eval_yolo(self.detector, dataloader, epoch, writer, logger, dataname=data_name)
        pred_bboxes = []
        pred_labels = []
        pred_scores = []
        gt_bboxes = []
        gt_labels = []
        gt_difficults = []

        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                utils.progress_bar(i, len(dataloader), 'Eva... ')
                image = sample['image'] #.to(opt.device)
                gt_bbox = sample['bboxes']
                labels = sample['labels']
                paths = sample['path']

                batch_bboxes, batch_labels, batch_scores = self.forward_test(image)

                # lables_b = copy.deepcopy(labels)
                # for label in lables_b:
                #     label += 1.
                # for b in range(len(image)):
                #     if len(gt_bbox[b]) == 0:
                #         return {}

                # gt_bbox = [bbox.to(opt.device).float() for bbox in gt_bbox]
                # lables_b = [label.to(opt.device).float() for label in lables_b]
                # image = list(im.to(opt.device) for im in image)

                # b = len(gt_bbox)

                # target = [{'boxes': gt_bbox[i], 'labels': lables_b[i].long()} for i in range(b)]
                # self.train()
                # loss_dict = self.detector(image, target)
                # self.eval()
                # loss = sum(l for l in loss_dict.values())
                # self.avg_meters.update({'loss': loss.item()})

                pred_bboxes.extend(batch_bboxes)
                pred_labels.extend(batch_labels)
                pred_scores.extend(batch_scores)

                for b in range(len(gt_bbox)):
                    gt_bboxes.append(gt_bbox[b].detach().cpu().numpy())
                    gt_labels.append(labels[b].int().detach().cpu().numpy())
                    gt_difficults.append(np.array([False] * len(gt_bbox[b])))

                if opt.vis:  # 可视化预测结果
                    image = image[0].unsqueeze(0)
                    img = tensor2im(image).copy()
                    # for x1, y1, x2, y2 in gt_bbox[0]:
                    #     cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 0), 2)  # 绿色的是gt

                    num = len(batch_scores[0])
                    visualize_boxes(image=img, boxes=batch_bboxes[0],
                             labels=batch_labels[0].astype(np.int32), probs=batch_scores[0], class_labels=config.DATA.CLASS_NAMES)

                    write_image(writer, f'{data_name}/{i}', 'image', img, epoch, 'HWC')

            write_meters_loss(writer, 'val', self.avg_meters, epoch)
            result = []
            # for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            for iou_thresh in [0.5]:
                AP = eval_detection_voc(
                    pred_bboxes,
                    pred_labels,
                    pred_scores,
                    gt_bboxes,
                    gt_labels,
                    gt_difficults=None,
                    iou_thresh=iou_thresh,
                    use_07_metric=False)

                APs = AP['ap']
                mAP = AP['map']
                result.append(mAP)
                if iou_thresh in [0.5, 0.75]:
                    logger.info(f'Eva({data_name}) epoch {epoch}, IoU: {iou_thresh}, APs: {str(APs[:10])}, mAP: {mAP}')

                write_loss(writer, f'val/{data_name}', 'mAP', mAP, epoch)

            logger.info(
                f'Eva({data_name}) epoch {epoch}, mean of (AP50-AP95): {sum(result)/len(result)}')

    def eval_pb(self, dataloader, epoch, writer, logger, data_name='val'):

        pred_bboxes = []
        pred_labels = []
        pred_scores = []
        gt_bboxes = []
        gt_labels = []
        gt_difficults = []

        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                utils.progress_bar(i, len(dataloader), 'Eva... ')
                image = sample['image'] #.to(opt.device)
                gt_bbox = sample['bboxes']
                labels = sample['labels']
                paths = sample['path']

                batch_bboxes, batch_labels, batch_scores = self.forward_test_pb(image)

                pred_bboxes.extend(batch_bboxes)
                pred_labels.extend(batch_labels)
                pred_scores.extend(batch_scores)

                for b in range(len(gt_bbox)):
                    gt_bboxes.append(gt_bbox[b].detach().cpu().numpy())
                    gt_labels.append(labels[b].int().detach().cpu().numpy())
                    gt_difficults.append(np.array([False] * len(gt_bbox[b])))

                if opt.vis:  # 可视化预测结果
                    image = image[0].unsqueeze(0)
                    img = tensor2im(image).copy()

                    num = len(batch_scores[0])
                    visualize_boxes_pb(image=img, boxes=batch_bboxes[0],
                             labels=batch_labels[0].astype(np.int32), probs=batch_scores[0], class_labels=config.DATA.CLASS_NAMES)

                    write_image(writer, f'{data_name}/{i}', 'image', img, epoch, 'HWC')

            write_meters_loss(writer, 'val', self.avg_meters, epoch)
            result = []
            # for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            for iou_thresh in [0.5]:
                AP = eval_detection_voc(
                    pred_bboxes,
                    pred_labels,
                    pred_scores,
                    gt_bboxes,
                    gt_labels,
                    gt_difficults=None,
                    iou_thresh=iou_thresh,
                    use_07_metric=False)

                APs = AP['ap']
                mAP = AP['map']
                result.append(mAP)
                if iou_thresh in [0.5, 0.75]:
                    logger.info(f'Eva({data_name}) epoch {epoch}, IoU: {iou_thresh}, APs: {str(APs[:10])}, mAP: {mAP}')

                write_loss(writer, f'val/{data_name}', 'mAP', mAP, epoch)

            logger.info(
                f'Eva({data_name}) epoch {epoch}, mean of (AP50-AP95): {sum(result)/len(result)}')


    def load(self, ckpt_path):
        if ckpt_path[-2:] != 'pt':
            return 0

        load_dict = {
            'detector': self.detector,
        }

        if opt.resume or 'RESUME' in self.config.MISC:
            load_dict.update({
                'optimizer': self.optimizer,
                'scheduler': self.scheduler,
            })
            utils.color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)
        else:
            utils.color_print('Load checkpoint from %s.' % ckpt_path, 3)

        ckpt_info = load_checkpoint(load_dict, ckpt_path, map_location=opt.device)

        s = torch.load(ckpt_path, map_location='cpu')
        if opt.resume or 'RESUME' in self.config.MISC:
            self.optimizer.load_state_dict(s['optimizer'])
            self.scheduler.step()

        epoch = ckpt_info.get('epoch', 0)

        return epoch

    def save(self, which_epoch, published=False):
        save_filename = f'{which_epoch}_{self.config.MODEL.NAME}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {
            'detector': self.detector,
            'epoch': which_epoch
        }

        if published:
            save_dict['epoch'] = 0
        else:
            save_dict['optimizer'] = self.optimizer
            save_dict['scheduler'] = self.scheduler

        save_checkpoint(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)
