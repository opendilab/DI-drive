import os
import sys
import time
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

from dataset import get_dataloader
from models import ImplicitSupervisedModel

config = dict(
    model=dict(hidden_size=1024, ),
    learner=dict(
        gpus=[0, 1, 2, 3],
        lr=0.0001,
        lr_step=10,
        lr_gamma=0.5,
        weight_decay=0.0001,
        printfreq=10,
        max_epoch=30,
        seed=2020,
        log_dir='./log',
    ),
    data=dict(
        dataset_dir='./dataset',
        batch_size=12,
        num_workers=8,
        crop_sky=True,
    ),
    resume=False,
)


class Loss(torch.nn.Module):
    '''
    Loss function for implicit affordances
    '''

    def __init__(self, weights=[1.0, 1.0, 10.0, 1.0, 1.0]):
        super(Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l2_loss = nn.MSELoss()
        self.weights = weights

    def forward(self, preds, gts):
        pred_seg, pred_is_junction, pred_tl_state, pred_tl_dis, pred_delta_yaw = preds
        gt_seg, gt_is_junction, gt_tl_state, gt_tl_dis, gt_delta_yaw = gts

        loss_seg = self.l2_loss(pred_seg, gt_seg) * self.weights[0]
        loss_is_junction = self.ce_loss(pred_is_junction, gt_is_junction) * self.weights[1]
        loss_tl_state = self.ce_loss(pred_tl_state, gt_tl_state) * self.weights[2]
        loss_tl_dis = self.ce_loss(pred_tl_dis, gt_tl_dis) * self.weights[3]
        loss_delta_yaw = self.l2_loss(pred_delta_yaw, gt_delta_yaw) * self.weights[4]

        loss = loss_seg + loss_is_junction + loss_tl_state + loss_tl_dis + loss_delta_yaw
        return loss, (
            loss.item(), loss_seg.item(), loss_is_junction.item(), loss_tl_state.item(), loss_tl_dis.item(),
            loss_delta_yaw.item()
        )


class lossMetric():
    '''
     Get the moving average of different kinds losses
    '''

    def __init__(self, keys):
        self.keys = keys
        self.metric = {}
        for key in keys:
            self.metric[key] = []

    def update(self, values):
        for i, key in enumerate(self.keys):
            self.metric[key].append(values[i])

    def reset(self):
        for key in self.keys:
            self.metric[key] = []

    def get_metric_dict(self):
        res = {}
        for key in self.keys:
            res[key] = np.mean(self.metric[key])
        return res


def train_or_eval(criterion, net, data, optim, is_train, config, epoch, writter):
    if is_train:
        desc = 'Train'
        net.train()
    else:
        desc = 'Val'
        net.eval()

    total = len(data)
    iterator_tqdm = tqdm.tqdm(data, desc=desc, total=total)
    iterator = enumerate(iterator_tqdm)

    epoch_iters = len(data)

    loss_metric = lossMetric(
        keys=['total_loss', 'loss_seg', 'loss_is_junction', 'loss_tl_state', 'loss_tl_dis', 'loss_delta_yaw']
    )

    for i, (rgb_images, gt_seg, gt_tl_state, gt_is_junction, gt_tl_dis, gt_delta_yaw) in iterator:
        rgb_images = rgb_images.cuda().float()
        gt_tl_state = gt_tl_state.long().cuda()
        gt_tl_dis = gt_tl_dis.long().cuda()
        gt_is_junction = gt_is_junction.long().cuda()
        gt_seg = gt_seg.cuda().float()
        gt_delta_yaw = gt_delta_yaw.cuda().float()

        pred_seg, pred_is_junction, pred_tl_state, pred_tl_dis, pred_delta_yaw = net(rgb_images)

        preds = pred_seg, pred_is_junction, pred_tl_state, pred_tl_dis, pred_delta_yaw
        gts = gt_seg, gt_is_junction, gt_tl_state, gt_tl_dis, gt_delta_yaw
        loss, loss_details = criterion(preds, gts)

        loss_metric.update(loss_details)

        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()

        should_log = False
        should_log |= (i != 0 and i % config['learner']['printfreq'] == 0)
        if should_log and is_train:
            metrics = loss_metric.get_metric_dict()
            loss_metric.reset()
            for key in metrics.keys():
                writter.add_scalar('train/' + key, metrics[key], i + epoch_iters * epoch)

        if not is_train:
            metrics = loss_metric.get_metric_dict()
            for key in metrics.keys():
                writter.add_scalar('val/' + key, metrics[key], i + epoch_iters * epoch)


def train(config):
    logging.basicConfig(
        format='%(asctime)-12s %(levelname)s: %(message)s',
        datefmt='%Y %b %d %H:%M:%S',
        level=logging.INFO,
    )

    os.makedirs(config['learner']['log_dir'], exist_ok=True)
    writter = SummaryWriter(log_dir=config['learner']['log_dir'])

    random.seed(config['learner']['seed'])
    np.random.seed(config['learner']['seed'])
    torch.manual_seed(config['learner']['seed'])

    data_train, data_val = get_dataloader(**config['data'])
    criterion = Loss()

    net = ImplicitSupervisedModel(4, 4, config['model']['hidden_size'], 6, 4, config['data']['crop_sky']).cuda()

    if config['resume']:
        log_dir = Path(config['learner']['log_dir'])
        checkpoints = list(log_dir.glob('model-*.pth.tar'))
        checkpoint = str(checkpoints[-1])
        if len(checkpoints) > 0:
            logging.info("=> lodding checkpoint '{}'".format(checkpoint))
            net.load_state_dict(torch.load(checkpoint))
        else:
            logging.warn("=> no checkpoint found at '{}'".format(checkpoint))

    net = torch.nn.DataParallel(net, device_ids=config['learner']['gpus'])

    torch.backends.cudnn.benchmark = True

    optim = torch.optim.Adam(
        net.parameters(), lr=config['learner']['lr'], weight_decay=config['learner']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=config['learner']['lr_step'], gamma=config['learner']['lr_gamma']
    )

    for epoch in tqdm.tqdm(range(config['learner']['max_epoch'] + 1), desc='Epoch'):
        train_or_eval(criterion, net, data_train, optim, True, config, epoch, writter)
        train_or_eval(criterion, net, data_val, None, False, config, epoch, writter)

        torch.save(net.state_dict(), str(Path(config['learner']['log_dir']) / ('model-%d.pth.tar' % epoch)))
        scheduler.step()


if __name__ == '__main__':
    train(config)
