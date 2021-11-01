import os
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from core.policy import LBCBirdviewPolicy
from core.data import LBCBirdViewDataset
from core.utils.simulator_utils.carla_utils import visualize_birdview


lbc_config = dict(
    exp_name='lbc_bev_train',
    data=dict(
        train=dict(
            root_dir='path_to_your_dataset',
            gap=5,
            n_step=5,
            crop_x_jitter=5,
            crop_y_jitter=0,
            angle_jitter=5,
        ),
        val=dict(
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
            batch_size=128,
            loss='l1',
            lr=1e-4,
        ),
    ),
)
main_config = EasyDict(lbc_config)


def train_or_eval(policy, loader, optim, is_train, config, is_first_epoch):
    if is_train:
        desc = 'Train'
    else:
        desc = 'Val'

    total = 10 if is_first_epoch else len(loader)
    iterator_tqdm = tqdm(loader, desc=desc, total=total)

    tick = time.time()
    policy.reset()

    for i, data in enumerate(iterator_tqdm):
        loss = policy.forward(data)

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss.backward()
            optim.step()

        tick = time.time()

        if is_first_epoch and i == 10:
            iterator_tqdm.close()
            break


def main(cfg):
    if cfg.policy.cudnn:
        torch.backends.cudnn.benchmark = True

    train_dataset = LBCBirdViewDataset(**cfg.data.train)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.policy.learn.batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True)

    lbc_policy = LBCBirdviewPolicy(cfg.policy)
    optim = torch.optim.Adam(lbc_policy._model.parameters(), lr=cfg.policy.learn.lr)

    for epoch in tqdm(range(cfg.policy.learn.epoches+1), desc='Epoch'):
        train_or_eval(lbc_policy.learn_mode, train_dataloader, optim, True, cfg, epoch == 0)


if __name__ == '__main__':
    main(main_config)
