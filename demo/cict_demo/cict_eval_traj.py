import os
import time

import sys
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
import torch
from torch.autograd import grad
import torchvision.transforms as transforms
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader

from core.data.cict_dataset import PathDataset
from demo.cict_demo.cict_model import ModelGRU
from core.utils.others.checkpoint_helper import get_latest_saved_checkpoint

eval_config = dict(
    NUMBER_OF_LOADING_WORKERS=1,
    IMG_HEIGHT=200,
    IMG_WIDTH=400,
    MAX_DIST=25.,
    MAX_T=1,
    IMG_STEP=1,
    START_EPISODE=37,
    END_EPISODE=40,
    BATCH_SIZE=1,
    PRED_LEN=10,
    COMMON=dict(folder='sample', exp='cict_traj', dataset_path='datasets'),
    GPU='0',
    SPEED_FACTOR=25.0,
    TRAIN_DATASET_NAME='cict_datasets_train',
    MODEL_TYPE='cict_traj',
    PREFIX='_preloads',
    EVAL=True,
    MODEL_CONFIGURATION=dict(input_dim=1, hidden_dim=256, out_dim=2),
    PRELOAD_MODEL_ALIAS=None,
    PRELOAD_MODEL_BATCH=None,
    PRELOAD_MODEL_CHECKPOINT=None,
    REMOVE=None,
)


def execute(cfg):
    gpu = cfg.GPU
    exp_batch = cfg.COMMON.folder
    exp_alias = cfg.COMMON.exp

    if gpu != '':
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

    full_dataset = os.path.join(cfg.COMMON.dataset_path, cfg.TRAIN_DATASET_NAME)

    ipm_transforms = [
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ]

    dataset = PathDataset(full_dataset, cfg, transform=ipm_transforms)

    print("Loaded dataset")

    data_loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUMBER_OF_LOADING_WORKERS
    )

    model = ModelGRU(cfg.MODEL_CONFIGURATION)

    if checkpoint_file is not None or cfg.PRELOAD_MODEL_ALIAS is not None:
        model.load_state_dict(checkpoint['state_dict'])

    if gpu != '':
        model.cuda()
    model.eval()

    print("Before the loss")

    criterion = torch.nn.MSELoss()

    print('Start to eval ...')
    loss_xy_list = []
    loss_vxy_list = []
    loss_axy_list = []
    for i, data in enumerate(data_loader):
        if gpu != '':
            ipms = data['ipms'].cuda()
            t = data['label_t'].cuda()
            cur_v = data['cur_v'].cuda()
            label_xy = data['label_xy'].cuda()
            label_vxy = data['label_vxy'].cuda()
            label_axy = data['label_axy'].cuda()
        else:
            ipms = data['ipms']
            t = data['label_t']
            cur_v = data['cur_v']
            label_xy = data['label_xy']
            label_vxy = data['label_vxy']
            label_axy = data['label_axy']
        ipms.requires_grad = True
        t.requires_grad = True

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
        loss_xy_list.append(loss_xy.data.cpu().numpy())
        loss_vxy_list.append(loss_vxy.data.cpu().numpy())
        loss_axy_list.append(loss_axy.data.cpu().numpy())

        name, ind, _ = dataset.get_info(i)
        save_dir = os.path.join('_logs', exp_batch, exp_alias, name)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        preds = torch.cat([pred_xy, pred_vxy, pred_axy], dim=0).detach().cpu().numpy()
        np.savetxt(os.path.join(save_dir, 'pred_%05d.txt' % (ind)), preds.reshape(3, -1), fmt='%f', delimiter=',')
        labels = torch.cat([label_xy, label_vxy, label_axy], dim=0).detach().cpu().numpy()
        np.savetxt(os.path.join(save_dir, 'label_%05d.txt' % (ind)), labels.reshape(3, -1), fmt='%f', delimiter=',')

        if loss_vxy.data > 0.5:
            print(
                "Episode: %s, id: %d, Loss_xy: %f  Loss_vxy: %f  Loss_axy: %f " %
                (name, ind, loss_xy.data, loss_vxy.data, loss_axy.data)
            )

    print(
        'total loss, xy: %f, vxy:%f, axy:%f' % (np.mean(loss_xy_list), np.mean(loss_vxy_list), np.mean(loss_axy_list))
    )


if __name__ == '__main__':
    cfg = EasyDict(eval_config)

    execute(cfg)
