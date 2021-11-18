import os
import torch
import cv2
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

from core.policy import LBCBirdviewPolicy
from core.data import LBCBirdViewDataset
from core.utils.simulator_utils.carla_utils import visualize_birdview
from core.utils.learner_utils.log_saver_utils import Experiment


lbc_config = dict(
    exp_name='lbc_bev_train',
    data=dict(
        train=dict(
            root_dir='lbc_train',
            gap=5,
            n_step=5,
            crop_x_jitter=5,
            crop_y_jitter=0,
            angle_jitter=5,
        ),
        val=dict(
            root_dir='lbc_val',
            crop_x_jitter=0,
            crop_y_jitter=0,
            angle_jitter=0,
        ),
    ),
    policy=dict(
        cudnn=True,
        cuda=True,
        learn=dict(
            epoches=1000,
            log_freq=1000,
            batch_size=128,
            loss='l1',
            lr=1e-4,
        ),
    ),
)
main_config = EasyDict(lbc_config)


def get_log_visualization(birdview, command, loss, locations, locations_pred, size=16):
    WHITE = [255, 255, 255]
    BLUE = [0, 0, 255]
    RED = [255, 0, 0]

    images = list()

    for i in range(min(birdview.shape[0], size)):
        loss_i = loss[i].sum()
        _bev = birdview[i].detach().cpu().numpy().copy()
        canvas = np.uint8(_bev * 255).copy()
        canvas = visualize_birdview(canvas)
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 10) for x in range(10+1)]

        def _write(text, i, j):
            cv2.putText(
                    canvas, text, (cols[j], rows[i]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        def _dot(i, j, color, radius=2):
            x, y = int(j), int(i)
            canvas[x-radius:x+radius+1, y-radius:y+radius+1] = color

        _command = {
                1: 'LEFT', 2: 'RIGHT',
                3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item()+1, '???')

        _dot(0, 0, WHITE)

        for x, y in locations[i]: _dot(x, y, BLUE)
        for x, y in (locations_pred[i] + 1) * (0.5 * 192): _dot(x, y, RED)

        _write('Command: %s' % _command, 1, 0)
        _write('Loss: %.2f' % loss[i].item(), 2, 0)

        images.append((loss[i].item(), canvas))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]


def train_or_eval(policy, loader, optim, is_train, config, is_first_epoch, log_saver):
    if is_train:
        desc = 'Train'
    else:
        desc = 'Val'

    total = 10 if is_first_epoch else len(loader)
    iterator_tqdm = tqdm(loader, desc=desc, total=total)

    tick = time.time()
    policy.reset()

    for i, data in enumerate(iterator_tqdm):
        res_dict = policy.forward(data)
        loss = res_dict['loss']
        loss_mean = loss.mean()

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()

        if i % config.policy.learn.log_freq == 0 or not is_train or is_first_epoch:
            metrics = dict()
            metrics['loss'] = loss_mean.item()

            images = get_log_visualization(
                data['birdview'], data['command'], loss, data['location'], res_dict['locations_pred'],
            )

            log_saver.scalar(is_train=is_train, loss_mean=loss_mean.item())
            log_saver.image(is_train=is_train, birdview=images)

        log_saver.scalar(is_train=is_train, fps=1.0/(time.time() - tick))

        tick = time.time()

        if is_first_epoch and i == 10:
            iterator_tqdm.close()
            break


def main(cfg):
    if cfg.policy.cudnn:
        torch.backends.cudnn.benchmark = True

    train_dataset = LBCBirdViewDataset(**cfg.data.train)
    val_dataset = LBCBirdViewDataset(**cfg.data.val)
    train_dataloader = DataLoader(
        train_dataset, cfg.policy.learn.batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, cfg.policy.learn.batch_size, num_workers=16, shuffle=False, drop_last=False, pin_memory=True
    )

    lbc_policy = LBCBirdviewPolicy(cfg.policy)
    log_saver = Experiment(log_dir='./log/{}/'.format(cfg.exp_name))
    optim = torch.optim.Adam(lbc_policy._model.parameters(), lr=cfg.policy.learn.lr)

    for epoch in tqdm(range(cfg.policy.learn.epoches+1), desc='Epoch'):
        train_or_eval(lbc_policy.learn_mode, train_dataloader, optim, True, cfg, epoch == 0, log_saver)
        train_or_eval(lbc_policy.learn_mode, val_dataloader, optim, False, cfg, epoch == 0, log_saver)

        if epoch in [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1000]:
            torch.save(
                    lbc_policy.learn_mode.state_dict(),
                    './log/{}/model-{}.th'.format(cfg.exp_name, epoch))

        log_saver.end_epoch()


if __name__ == '__main__':
    main(main_config)
