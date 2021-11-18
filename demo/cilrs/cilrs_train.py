from collections import defaultdict
import os
import numpy as np
from ding.utils.data.collate_fn import default_collate, default_decollate
from easydict import EasyDict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import Adam

from core.policy import CILRSPolicy
from core.data import CILRSDataset

config = dict(
    exp_name='cilrs_train',
    policy=dict(
        cuda=True,
        cudnn=True,
        resume=False,
        ckpt_path=None,
        model=dict(
            num_branch=4,
        ),
        learn=dict(
            epoches=200,
            batch_size=128,
            loss='l1',
            lr=1e-4,
            speed_weight=0.05,
            control_weights=[0.5, 0.45, 0.05],
        ),
        eval=dict(
            eval_freq=10,
        )
    ),
    data=dict(
        train=dict(
            root_dir='./datasets_train/cilrs_datasets_train',
            preloads='./_preloads/cilrs_datasets_train.npy',
            transform=True,
        ),
        val=dict(
            root_dir='./datasets_train/cilrs_datasets_val',
            preloads='./_preloads/cilrs_datasets_val.npy',
            transform=True,
        ),
    )
)
main_config = EasyDict(config)


def train(policy, optimizer, loader, tb_logger=None, start_iter=0):
    loss_epoch = defaultdict(list)
    iter_num = start_iter
    policy.reset()

    for data in tqdm(loader):
        log_vars = policy.forward(data)
        optimizer.zero_grad()
        total_loss = log_vars['total_loss']
        total_loss.backward()
        optimizer.step()
        log_vars['cur_lr'] = optimizer.defaults['lr']
        for k, v in log_vars.items():
            loss_epoch[k] += [log_vars[k].item()]
            if iter_num % 50 == 0 and tb_logger is not None:
                tb_logger.add_scalar("train_iter/" + k, v, iter_num)
        iter_num += 1
    loss_epoch = {k: np.mean(v) for k, v in loss_epoch.items()}
    return iter_num, loss_epoch


def validate(policy, loader, tb_logger=None, epoch=0):
    loss_epoch = defaultdict(list)
    policy.reset()
    for data in tqdm(loader):
        with torch.no_grad():
            log_vars = policy.forward(data)
        for k in list(log_vars.keys()):
            loss_epoch[k] += [log_vars[k]]
    loss_epoch = {k: np.mean(v) for k, v in loss_epoch.items()}
    if tb_logger is not None:
        for k, v in loss_epoch.items():
            tb_logger.add_scalar("validate_epoch/" + k, v, epoch)
    return loss_epoch


def save_ckpt(state, name=None, exp_name=''):
    os.makedirs('checkpoints/' + exp_name, exist_ok=True)
    ckpt_path = 'checkpoints/{}/{}_ckpt.pth'.format(exp_name, name)
    torch.save(state, ckpt_path)


def load_best_ckpt(policy, optimizer=None, root_dir='checkpoints', exp_name='', ckpt_path=None):
    ckpt_dir = os.path.join(root_dir, exp_name)
    assert os.path.isdir(ckpt_dir), ckpt_dir
    files = os.listdir(ckpt_dir)
    assert files, 'No ckpt files found'

    if ckpt_path and ckpt_path in files:
        pass
    elif os.path.exists(os.path.join(ckpt_dir, 'best_ckpt.pth')):
        ckpt_path = 'best_ckpt.pth'
    else:
        ckpt_path = sorted(files)[-1]
    print('Load ckpt:', ckpt_path)
    state_dict = torch.load(os.path.join(ckpt_dir, ckpt_path))
    policy.load_state_dict(state_dict)
    if 'optimizer' in state_dict:
        optimizer.load_state_dict(state_dict['optimizer'])
    epoch = state_dict['epoch']
    iterations = state_dict['iterations']
    best_loss = state_dict['best_loss']
    return epoch, iterations, best_loss


def main(cfg):
    if cfg.policy.cudnn:
        torch.backends.cudnn.benchmark = True

    train_dataset = CILRSDataset(**cfg.data.train)
    val_dataset = CILRSDataset(**cfg.data.val)
    train_loader = DataLoader(train_dataset, cfg.policy.learn.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, cfg.policy.learn.batch_size, num_workers=8)

    cilrs_policy = CILRSPolicy(cfg.policy)
    optimizer = Adam(cilrs_policy._model.parameters(), cfg.policy.learn.lr)
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    iterations = 0
    best_loss = 1e8
    start_epoch = 0

    if cfg.policy.resume:
        start_epoch, iterations, best_loss = load_best_ckpt(
            cilrs_policy.learn_mode, optimizer, exp_name=cfg.exp_name, ckpt_path=cfg.policy.ckpt_path
        )

    for epoch in range(start_epoch, cfg.policy.learn.epoches):
        iter_num, loss = train(cilrs_policy.learn_mode, optimizer, train_loader, tb_logger, iterations)
        iterations = iter_num
        tqdm.write(
            f"Epoch {epoch:03d}, Iter {iter_num:06d}: Total: {loss['total_loss']:2.5f}" +
            f" Speed: {loss['speed_loss']:2.5f} Str: {loss['steer_loss']:2.5f}" +
            f" Thr: {loss['throttle_loss']:2.5f} Brk: {loss['brake_loss']:2.5f}"
        )
        if epoch % cfg.policy.eval.eval_freq == 0:
            loss_dict = validate(cilrs_policy.learn_mode, val_loader, tb_logger, iterations)
            total_loss = loss_dict['total_loss']
            tqdm.write(f"Validate Total: {total_loss:2.5f}")
            state_dict = cilrs_policy.learn_mode.state_dict()
            state_dict['optimizer'] = optimizer.state_dict()
            state_dict['epoch'] = epoch
            state_dict['iterations'] = iterations
            state_dict['best_loss'] = best_loss
            if total_loss < best_loss and epoch > 0:
                tqdm.write("Best Validation Loss!")
                best_loss = total_loss
                state_dict['best_loss'] = best_loss
                save_ckpt(state_dict, 'best', cfg.exp_name)
            save_ckpt(state_dict, '{:05d}'.format(epoch), cfg.exp_name)


if __name__ == '__main__':
    main(main_config)
