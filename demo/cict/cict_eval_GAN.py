import os

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from core.data.cict_dataset import CictDataset
from core.utils.others.checkpoint_helper import get_latest_saved_checkpoint
from demo.cict_demo.cict_model import GeneratorUNet
from demo.cict_demo.post import Sensor, params, InversePerspectiveMapping

eval_config = dict(
    NUMBER_OF_LOADING_WORKERS=1,
    IMG_HEIGHT=128,
    IMG_WIDTH=256,
    SENSORS=dict(rgb=[3, 360, 640]),
    DEST=1,
    START_EPISODE=37,
    END_EPISODE=40,
    BATCH_SIZE=1,
    COMMON=dict(folder='sample', exp='cict_GAN', dataset_path='datasets'),
    GPU='0',
    TRAIN_DATASET_NAME='cict_datasets_train',
    MODEL_TYPE='cict_GAN',
    PREFIX='_preloads',
    UNPAIRED=False,
    MODEL_CONFIGURATION=dict(
        generator=dict(
            down_channels=[6, 64, 128, 256, 512, 512, 512, 512],
            up_channels=[0, 512, 512, 512, 256, 128, 64],
            kernel_size=4,
            stride=2,
            padding=1,
            down_norm=[False, True, True, True, True, True, False],
            up_norm=[True, True, True, True, True, True],
            down_dropout=[0, 0, 0, 0.5, 0.5, 0.5, 0.5],
            up_dropout=[0.5, 0.5, 0.5, 0, 0, 0],
            final_channels=1,
            num_branches=1,
        )
    ),
    GAN_LOSS_FUNCTION='MSE',
    PIXEL_LOSS_FUNCTION='L1',
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

    pm_transforms = [
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ]

    img_transforms = [
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dest_transforms = [
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataset = CictDataset(
        full_dataset, cfg, img_transform=img_transforms, dest_transform=dest_transforms, pm_transform=pm_transforms
    )

    print("Loaded dataset")

    data_loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUMBER_OF_LOADING_WORKERS
    )

    generator = GeneratorUNet(cfg.MODEL_CONFIGURATION['generator'])

    if checkpoint_file is not None or cfg.PRELOAD_MODEL_ALIAS is not None:
        generator.load_state_dict(checkpoint['state_dict_G'])

    if gpu != '':
        generator.cuda()
    generator.eval()
    print("Before the loss")

    criterion_pixel = torch.nn.L1Loss()

    print('Start to eval ...')

    for i, data in enumerate(data_loader):
        img = data['rgb']
        dest = data['dest']
        pm = data['pm']
        command = data['command']
        input = torch.cat([img, dest], dim=1)

        if gpu != '':
            input = input.cuda()
            pm = pm.cuda()

        pm_fake = generator(input, command)
        loss_pixel = criterion_pixel(pm_fake, pm)

        name, ind = dataset.get_info(i)
        save_dir = os.path.join(full_dataset, name, 'pm')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_image(pm_fake.data, os.path.join(save_dir, 'fake_%05d.png' % ind), nrow=1, normalize=True)
        sensor = Sensor(params.sensor_config['rgb'])
        inverse_perspective_mapping = InversePerspectiveMapping(params, sensor)

        lidar = np.load(os.path.join(full_dataset, name, "lidar_%05d.npy" % ind))
        pm = pm_fake[0, 0].detach().cpu().numpy()
        pm = (pm + 1) / 2 * 255
        pm = cv2.resize(pm, (params.sensor_config['rgb']['img_width'], params.sensor_config['rgb']['img_height']))

        ipm = inverse_perspective_mapping.getIPM(pm)
        img = inverse_perspective_mapping.get_cost_map(ipm, lidar)
        cv2.imwrite(os.path.join(full_dataset, name, 'ipm/pred_%05d.png' % ind), img)
        print("%s ID:%05d Loss_pixel: %f Command:%d" % (name, ind, loss_pixel.data, command.data))


if __name__ == '__main__':
    cfg = EasyDict(eval_config)

    execute(cfg)
