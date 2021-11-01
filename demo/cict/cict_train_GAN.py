import os
import time
import json
import glob

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from easydict import EasyDict
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms.transforms import Grayscale

from core.data.cict_dataset import CictDataset
from demo.cict_demo.cict_model import GeneratorUNet, Discriminator
from core.utils.learner_utils.loss_utils import Loss
#from core.utils.learner_utils.optim_utils import adjust_learning_rate_auto
from core.utils.others.checkpoint_helper import is_ready_to_save, get_latest_saved_checkpoint
from core.utils.others.general_helper import create_log_folder, create_exp_path, erase_logs

train_config = dict(
    NUMBER_OF_LOADING_WORKERS=4,
    IMG_HEIGHT=128,
    IMG_WIDTH=256,
    SENSORS=dict(rgb=[3, 360, 640]),
    DEST=0,  # choose bird-view destination (0) or camera-view destination (1)
    START_EPISODE=0,  # set which episodes for training
    END_EPISODE=37,
    BATCH_SIZE=32,
    COMMON=dict(folder='sample', exp='cict_GAN', dataset_path='datasets'),
    GPU='0',
    SAVE_INTERVAL=1000,
    MAX_CKPT_SAVE_NUM=40,
    N_EPOCHS=60,
    SPEED_FACTOR=25.0,
    TRAIN_DATASET_NAME='cict_datasets_train',
    MODEL_TYPE='cict_GAN',
    PREFIX='_preloads',
    UNPAIRED=False,
    GAN=False,
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
        ),
        discriminator=dict(
            channels=[7, 64, 128, 256, 512],
            kernel_size=4,
            stride=2,
            padding=1,
            norm=[False, True, True, True],
            dropout=[0, 0, 0, 0]
        )
    ),
    PRE_TRAINED=False,
    LEARNING_RATE=0.0003,
    BETA1=0.5,
    BETA2=0.999,
    GAN_LOSS_FUNCTION='MSE',
    PIXEL_LOSS_FUNCTION='L1',
    PIXEL_LOSS_WEIGHT=2,
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
        best_loss_G = checkpoint['best_loss_G']
        best_loss_iter_G = checkpoint['best_loss_iter_G']
        best_loss_D = checkpoint['best_loss_D']
        best_loss_iter_D = checkpoint['best_loss_iter_D']
    else:
        if not os.path.exists(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints')):
            os.mkdir(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints'))

        iteration = 0
        best_loss_G = 10000.0
        best_loss_iter_G = 0
        best_loss_D = 10000.0
        best_loss_iter_D = 0

    write_params(os.path.join('_logs', exp_batch, exp_alias), train_config)
    full_dataset = os.path.join(cfg.COMMON.dataset_path, cfg.TRAIN_DATASET_NAME)

    pm_transforms = [
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ]

    img_transforms = [
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH), Image.BICUBIC),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dest_transforms = [
        transforms.Resize((cfg.IMG_HEIGHT, cfg.IMG_WIDTH), Image.BICUBIC),
        transforms.RandomRotation(15, resample=Image.BICUBIC, expand=False),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataset = CictDataset(
        full_dataset, cfg, img_transform=img_transforms, dest_transform=dest_transforms, pm_transform=pm_transforms
    )

    print("Loaded dataset")
    sample_weights = dataset.sample_weights()
    print(len(sample_weights), len(dataset))
    sampler = WeightedRandomSampler(sample_weights, len(dataset))

    data_loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=cfg.NUMBER_OF_LOADING_WORKERS
    )

    generator = GeneratorUNet(cfg.MODEL_CONFIGURATION['generator'])
    discriminator = Discriminator(cfg.MODEL_CONFIGURATION['discriminator'])

    generator.cuda()
    discriminator.cuda()

    optimizer_G = optim.Adam(generator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA1, cfg.BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA1, cfg.BETA2))

    if checkpoint_file is not None or cfg.PRELOAD_MODEL_ALIAS is not None:
        generator.load_state_dict(checkpoint['state_dict_G'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        discriminator.load_state_dict(checkpoint['state_dict_D'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        accumulated_time = checkpoint['total_time']
        loss_window = checkpoint['loss_window']
    else:  # We accumulate iteration time and keep the average speed
        accumulated_time = 0
        loss_window = []

    print("Before the loss")

    criterion_GAN = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()

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
            img = data['rgb']
            dest = data['dest']
            pm = data['pm']
            command = data['command']

            valid = torch.ones(img.size(0), 1, cfg.IMG_HEIGHT // 16, cfg.IMG_WIDTH // 16).cuda()
            fake = torch.zeros(img.size(0), 1, cfg.IMG_HEIGHT // 16, cfg.IMG_WIDTH // 16).cuda()

            input = torch.cat([img, dest], dim=1).cuda()

            pm = pm.cuda()

            generator.zero_grad()

            pm_fake = generator(input, command)
            #print(input.shape, pm_fake.shape)
            pred_fake = discriminator(pm_fake, input)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixel(pm_fake, pm)
            if cfg.UNPAIRED:
                fake_dest = data['fake_dest']
                fake_input = torch.cat([img, fake_dest], dim=1).cuda()
                pm_fake2 = generator(fake_input, command)
                pred_fake2 = discriminator(pm_fake2, fake_input)
                loss_GAN2 = criterion_GAN(pred_fake2, valid)
                loss_G = 0.5 * (loss_GAN + loss_GAN2) + cfg.PIXEL_LOSS_WEIGHT * loss_pixel
            else:
                if not cfg.GAN:
                    loss_G = cfg.PIXEL_LOSS_WEIGHT * loss_pixel
                else:
                    cfg.PIXEL_LOSS_WEIGHT * loss_pixel + loss_GAN
            loss_G.backward()
            torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value=20)
            optimizer_G.step()

            if cfg.GAN:
                discriminator.zero_grad()
                pred_real = discriminator(pm, input)
                loss_real = criterion_GAN(pred_real, valid)
                pred_fake = discriminator(pm_fake.detach(), input)
                loss_fake = criterion_GAN(pred_fake, fake)
                if cfg.UNPAIRED:
                    pred_fake2 = discriminator(pm_fake2.detach(), fake_input)
                    loss_fake2 = criterion_GAN(pred_fake2, fake)
                    loss_D = 0.5 * (loss_real + 0.5 * (loss_fake + loss_fake2))
                else:
                    loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                torch.nn.utils.clip_grad_value_(discriminator.parameters(), clip_value=20)
                optimizer_D.step()
            else:
                loss_D = torch.FloatTensor([0]).cuda()

            if iteration % cfg.SAVE_INTERVAL == 0:
                remove_old_ckpt(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints'), cfg)
                state = {
                    'iteration': iteration,
                    'state_dict_G': generator.state_dict(),
                    'state_dict_D': discriminator.state_dict(),
                    'best_loss_G': best_loss_G,
                    'best_loss_D': best_loss_D,
                    'total_time': accumulated_time,
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'best_loss_iter_G': best_loss_iter_G,
                    'best_loss_iter_D': best_loss_iter_D,
                    'loss_window': loss_window
                }
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias, 'checkpoints', str(iteration) + '.pth'))

            if loss_G.data < best_loss_G:
                best_loss_G = loss_G.data.tolist()
                best_loss_iter_G = iteration

            if loss_D.data < best_loss_D:
                best_loss_D = loss_D.data
                best_loss_iter_D = iteration

            accumulated_time += time.time() - capture_time

            loss_dict = {
                'loss_G': loss_G.data.tolist(),
                'loss_D': loss_D.data.tolist(),
                'loss_GAN': loss_GAN.data.tolist(),
                'loss_pixel': loss_pixel.data.tolist(),
            }

            loss_window.append(loss_dict)
            print(
                "Iteration: %d  Loss_pixel: %f  Loss_GAN: %f  Loss_G: %f  Loss_D: %f" %
                (iteration, loss_pixel.data, loss_GAN.data, loss_G.data, loss_D.data)
            )


if __name__ == '__main__':
    cfg = EasyDict(train_config)
    create_log_folder(cfg.COMMON.folder)
    erase_logs(cfg.COMMON.folder)

    create_exp_path(cfg.COMMON.folder, cfg.COMMON.exp)

    execute(cfg)
