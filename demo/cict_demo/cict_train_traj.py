import os
import time
import json
import glob

import torch
import torch.optim as optim
from torch.autograd import grad
import torchvision.transforms as transforms
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image

from core.data.cict_dataset import PathDataset
from demo.cict_demo.cict_model import ModelGRU
#from core.utils.learner_utils.loss_utils import Loss
#from core.utils.learner_utils.optim_utils import adjust_learning_rate_auto
from core.utils.others.checkpoint_helper import is_ready_to_save, get_latest_saved_checkpoint
from core.utils.others.general_helper import create_log_folder, create_exp_path, erase_logs

train_config = dict(
    NUMBER_OF_LOADING_WORKERS=4,
    IMG_HEIGHT=200,
    IMG_WIDTH=400,
    MAX_DIST=25.,
    MAX_T=1,
    IMG_STEP=1,
    PRED_LEN=10,
    START_EPISODE=0,
    END_EPISODE=37,
    BATCH_SIZE=64,
    COMMON=dict(folder='sample', exp='cict_traj', dataset_path='datasets'),
    GPU='0',
    EVAL=False,
    SAVE_INTERVAL=1000,
    MAX_CKPT_SAVE_NUM=40,
    N_EPOCHS=80,
    SPEED_FACTOR=25.0,
    TRAIN_DATASET_NAME='cict_datasets_train',
    MODEL_TYPE='cict_traj',
    PREFIX='_preloads',
    MODEL_CONFIGURATION=dict(input_dim=1, hidden_dim=256, out_dim=2),
    PRE_TRAINED=False,
    LEARNING_RATE=3e-4,
    WEIGHT_DECAY=5e-4,
    VELOCITY_LOSS_WEIGHT=1,
    ACCELERATION_LOSS_WEIGHT=0.1,
    PRELOAD_MODEL_ALIAS=None,
    PRELOAD_MODEL_BATCH=None,
    PRELOAD_MODEL_CHECKPOINT=None,
    REMOVE=None,
)


def write_params(log_path, config):

    with open(os.path.join(log_path, 'params.json'), 'w+') as f:
        json.dump('# Params', f)
        json.dump(config, f)


def remove_old_ckpt(ckpt_path, cfg):
    # get infos of all saved checkpoints
    ckpt_list = glob.glob(os.path.join(ckpt_path, '*.pth'))
    # sort checkpoints by saving time
    ckpt_list.sort(key=os.path.getmtime)
    # remove surplus ckpt file if the number is larger than max_ckpt_save_num
    if len(ckpt_list) >= cfg.MAX_CKPT_SAVE_NUM:
        for cur_file_idx in range(0, len(ckpt_list) - cfg.MAX_CKPT_SAVE_NUM + 1):
            os.remove(ckpt_list[cur_file_idx])


def execute(cfg):
    gpu = cfg.GPU
    exp_batch = cfg.COMMON.folder
    exp_alias = cfg.COMMON.exp

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if cfg.PRELOAD_MODEL_ALIAS is not None:
        checkpoint = torch.load(
            os.path.join(
                '_logs', cfg.PRELOAD_MODEL_BATCH, cfg.PRELOAD_MODEL_ALIAS, 'checkpoints',
                str(cfg.PRELOAD_MODEL_CHECKPOINT) + '.pth'
            )
        )

    checkpoint_file = get_latest_saved_checkpoint(exp_batch, exp_alias)
    if checkpoint_file is not None:
        checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints', checkpoint_file))
        iteration = checkpoint['iteration']
        best_loss = checkpoint['best_loss']
        best_loss_iter = checkpoint['best_loss_iter']

    else:
        if not os.path.exists(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints')):
            os.mkdir(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints'))

        iteration = 0
        best_loss = 10000.0
        best_loss_iter = 0

    write_params(os.path.join('_logs', exp_batch, exp_alias), train_config)
    full_dataset = os.path.join(cfg.COMMON.dataset_path, cfg.TRAIN_DATASET_NAME)

    ipm_transforms = [
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ]

    dataset = PathDataset(full_dataset, cfg, transform=ipm_transforms)

    print("Loaded dataset")

    sample_weights = dataset.sample_weights()
    print(len(sample_weights), len(dataset))
    sampler = WeightedRandomSampler(sample_weights, len(dataset))

    data_loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=cfg.NUMBER_OF_LOADING_WORKERS
    )

    model = ModelGRU(cfg.MODEL_CONFIGURATION)

    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)

    if checkpoint_file is not None or cfg.PRELOAD_MODEL_ALIAS is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        accumulated_time = checkpoint['total_time']
        loss_window = checkpoint['loss_window']
    else:  # We accumulate iteration time and keep the average speed
        accumulated_time = 0
        loss_window = []

    print("Before the loss")

    criterion = torch.nn.MSELoss()

    print('Start to train ...')

    iteration = 0
    for epoch in range(cfg.N_EPOCHS):
        for data in data_loader:
            iteration += 1

            #if iteration % 1000 == 0:
            #    adjust_learning_rate_auto(
            #        optimizer, loss_window, cfg.LEARNING_RATE, cfg.LEARNING_RATE_THRESHOLD,
            #        cfg.LEARNING_RATE_DECAY_LEVEL
            #    )

            capture_time = time.time()
            ipms = data['ipms'].cuda()
            t = data['label_t'].cuda()
            cur_v = data['cur_v'].cuda()
            label_xy = data['label_xy'].cuda()
            label_vxy = data['label_vxy'].cuda()
            label_axy = data['label_axy'].cuda()
            ipms.requires_grad = True
            t.requires_grad = True

            model.zero_grad()
            pred_xy = model(ipms, t, cur_v)

            pred_vx = grad(pred_xy[:, :, 0].sum(), t, create_graph=True)[0] * (cfg.MAX_DIST / cfg.MAX_T)
            pred_vy = grad(pred_xy[:, :, 1].sum(), t, create_graph=True)[0] * (cfg.MAX_DIST / cfg.MAX_T)
            #print(pred_vx.shape)
            pred_vxy = torch.cat([pred_vx.unsqueeze(-1), pred_vy.unsqueeze(-1)], dim=-1)

            pred_ax = grad(pred_vx.sum(), t, create_graph=True)[0] / cfg.MAX_T
            pred_ay = grad(pred_vy.sum(), t, create_graph=True)[0] / cfg.MAX_T
            #print(pred_ax.shape)
            pred_axy = torch.cat([pred_ax.unsqueeze(-1), pred_ay.unsqueeze(-1)], dim=-1)

            loss_xy = criterion(pred_xy, label_xy)
            loss_vxy = criterion(pred_vxy, label_vxy)
            loss_axy = criterion(pred_axy, label_axy)
            loss = loss_xy + cfg.VELOCITY_LOSS_WEIGHT * loss_vxy + cfg.ACCELERATION_LOSS_WEIGHT * loss_axy
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
            optimizer.step()

            if iteration % cfg.SAVE_INTERVAL == 0:
                remove_old_ckpt(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints'), cfg)
                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'optimizer': optimizer.state_dict(),
                    'best_loss_iter': best_loss_iter,
                    'loss_window': loss_window
                }
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias, 'checkpoints', str(iteration) + '.pth'))

            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            accumulated_time += time.time() - capture_time

            loss_dict = {
                'loss_xy': loss_xy.data.tolist(),
                'loss_vxy': loss_vxy.data.tolist(),
                'loss_axy': loss_axy.data.tolist(),
                'loss': loss.data.tolist(),
            }

            loss_window.append(loss_dict)
            print(
                "Iteration: %d  Loss_xy: %f  Loss_vxy: %f  Loss_axy: %f  Loss: %f" %
                (iteration, loss_xy.data, loss_vxy.data, loss_axy.data, loss.data)
            )


if __name__ == '__main__':
    cfg = EasyDict(train_config)
    create_log_folder(cfg.COMMON.folder)
    erase_logs(cfg.COMMON.folder)

    create_exp_path(cfg.COMMON.folder, cfg.COMMON.exp)

    execute(cfg)
