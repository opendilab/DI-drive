import os
import torch
import cv2
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
from pathlib import Path
import glob

from core.policy import LBCBirdviewPolicy, LBCImagePolicy
from core.data import LBCImageDataset
from core.utils.simulator_utils.carla_utils import visualize_birdview
from core.utils.learner_utils.log_saver_utils import Experiment

lbc_config = dict(
    exp_name='lbc_img_train_p1',
    data=dict(
        train=dict(
            root_dir='lbc_datasets_train',
            gap=5,
            n_step=5,
            batch_aug=1,
            augment_strategy='super_hard'
        ),
        val=dict(
            root_dir='lbc_datasets_val',
            gap=5,
            n_step=5,
            batch_aug=1,
            augment_strategy='super_hard'
        ),
    ),
    policy=dict(
        cudnn=True,
        cuda=True,
        model=dict(
            backbone='resnet34',
            pretrained=True,
            all_branch=True,
            resume=True,
        ),
        learn=dict(
            epoches=256,
            log_freq=1000,
            batch_size=24,
            loss='l1',
            lr=1e-4,
        ),
        gap=5,
        dt=0.1,
        camera_args=dict(
            w=384,
            h=160,
            fov=90,
            world_y=1.4,
            fixed_offset=4.0,
            n_step=5,
            crop_size=192,
            pixels_per_meter=5,
        ),
    ),
    teacher_policy=dict(
        ckpt_path='model-256.th',
        model=dict(all_branch=True, ),
    ),
    speed_noise=0.1,
)
main_config = EasyDict(lbc_config)

CROP_SIZE = main_config.policy.camera_args.crop_size
BATCH_AUG = main_config.data.train.batch_aug


class CoordConverter():

    def __init__(
        self,
        w=384,
        h=160,
        fov=90,
        world_y=1.4,
        fixed_offset=2.0,
        pixels_per_meter=5,
        crop_size=192,
        device='cuda',
        **kwargss
    ):
        self._img_size = torch.FloatTensor([w, h]).to(device)

        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset
        self._pixels_per_meter = pixels_per_meter
        self._crop_size = crop_size

    def __call__(self, camera_locations):
        camera_locations = (camera_locations + 1) * self._img_size / 2
        w, h = self._img_size

        cx, cy = w / 2, h / 2

        f = w / (2 * np.tan(self._fov * np.pi / 360))

        xt = (camera_locations[..., 0] - cx) / f
        yt = (camera_locations[..., 1] - cy) / f

        world_z = self._world_y / yt
        world_x = world_z * xt

        map_output = torch.stack([world_x, world_z], dim=-1)

        map_output *= self._pixels_per_meter
        map_output[..., 1] = self._crop_size - map_output[..., 1]
        map_output[..., 0] += self._crop_size / 2
        map_output[..., 1] += self._fixed_offset * self._pixels_per_meter

        return map_output


class LocationLoss(torch.nn.Module):

    def __init__(self, crop_size=192, **kwargs):
        super().__init__()
        self._crop_size = crop_size

    def forward(self, pred_locations, teac_locations):
        pred_locations = pred_locations / (0.5 * self._crop_size) - 1

        return torch.mean(torch.abs(pred_locations - teac_locations), dim=(1, 2, 3))


def get_log_visualization(rgb_image, birdview, command, loss, pred_locations, _pred_locations, _teac_locations, size=8):

    WHITE = [255, 255, 255]
    BLUE = [0, 0, 255]
    RED = [255, 0, 0]

    images = list()

    command_index = [i.item() - 1 for i in command]
    command = torch.FloatTensor(torch.eye(4))[command_index]

    for i in range(min(birdview.shape[0], size)):
        loss_i = loss[i].sum()

        _bev = birdview[i].detach().cpu().numpy().copy()
        canvas = np.uint8(_bev * 255).copy()
        canvas = visualize_birdview(canvas)

        _rgb = rgb_image[i].detach().cpu().numpy().copy()
        rgb = np.uint8(_rgb.transpose(1, 2, 0) * 255).copy()

        rows = [x * (canvas.shape[0] // 10) for x in range(10 + 1)]
        cols = [x * (canvas.shape[1] // 10) for x in range(10 + 1)]

        def _write(text, i, j):
            cv2.putText(canvas, text, (cols[j], rows[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        def _dot(_canvas, i, j, color, radius=2):
            x, y = int(j), int(i)
            _canvas[x - radius:x + radius + 1, y - radius:y + radius + 1] = color

        def _stick_together(a, b):
            h = min(a.shape[0], b.shape[0])

            r1 = h / a.shape[0]
            r2 = h / b.shape[0]

            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

            return np.concatenate([a, b], 1)

        _command = {1: 'LEFT', 2: 'RIGHT', 3: 'STRAIGHT', 4: 'FOLLOW'}.get(torch.argmax(command[i]).item() + 1, '???')

        _dot(canvas, 0, 0, WHITE)

        for x, y in (_teac_locations[i] + 1) * (0.5 * CROP_SIZE):
            _dot(canvas, x, y, BLUE)
        for x, y in _pred_locations[i]:
            _dot(rgb, x, y, RED)
        for x, y in pred_locations[i]:
            _dot(canvas, x, y, RED)

        _write('Command: %s' % _command, 1, 0)
        _write('Loss: %.2f' % loss[i].item(), 2, 0)

        images.append((loss[i].item(), _stick_together(rgb, canvas)))

    return [x[1] for x in images]


def repeat(a, repeats, dim=0):
    """
    Substitute for numpy's repeat function. Taken from https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/2
    torch.repeat([1,2,3], 2) --> [1, 2, 3, 1, 2, 3]
    np.repeat([1,2,3], repeats=2, axis=0) --> [1, 1, 2, 2, 3, 3]

    :param a: tensor
    :param repeats: number of repeats
    :param dim: dimension where to repeat
    :return: tensor with repitions
    """

    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = repeats
    a = a.repeat(*(repeat_idx))
    if a.is_cuda:  # use cuda-device if input was on cuda device already
        order_index = torch.cuda.LongTensor(
            torch.cat([init_dim * torch.arange(repeats, device=a.device) + i for i in range(init_dim)])
        )
    else:
        order_index = torch.LongTensor(torch.cat([init_dim * torch.arange(repeats) + i for i in range(init_dim)]))

    return torch.index_select(a, dim, order_index)


def train_or_eval(
    coord_converter, criterion, policy, teacher_policy, loader, optim, is_train, config, is_first_epoch, log_saver
):
    if is_train:
        desc = 'Train'
    else:
        desc = 'Val'

    total = 10 if is_first_epoch else len(loader)
    iterator_tqdm = tqdm(loader, desc=desc, total=total)

    tick = time.time()
    policy.reset()
    teacher_policy.reset()

    noise = main_config.speed_noise

    import torch.distributions as tdist
    noiser = tdist.Normal(torch.tensor(0.0), torch.tensor(noise))

    for i, data in enumerate(iterator_tqdm):

        rgb_image = data['rgb']
        birdview = data['birdview']
        command = data['command']
        speed = data['speed']

        if is_train and config['speed_noise'] > 0:
            speed += noiser.sample(speed.size()).to(speed.device)
            speed = torch.clamp(speed, 0, 10)

        if len(rgb_image.size()) > 4:
            B, batch_aug, c, h, w = rgb_image.size()
            rgb_image = rgb_image.view(B * batch_aug, c, h, w)
            birdview = repeat(birdview, batch_aug)
            command = repeat(command, batch_aug)
            speed = repeat(speed, batch_aug)

        rgb_data = {'rgb': rgb_image, 'command': command, 'location': data['location'], 'speed': speed}
        birdview_data = {'birdview': birdview, 'command': command, 'location': data['location'], 'speed': speed}

        with torch.no_grad():
            teacher_res_dict = teacher_policy.forward(birdview_data)
        res_dict = policy.forward(rgb_data)

        _pred_location = res_dict['locations_pred']
        _pred_locations = res_dict['all_branch_locations_pred']
        _teac_location = teacher_res_dict['locations_pred']
        _teac_locations = teacher_res_dict['all_branch_locations_pred']

        pred_location = coord_converter(_pred_location)
        pred_locations = coord_converter(_pred_locations)

        loss = criterion(pred_locations, _teac_locations)
        loss_mean = loss.mean()

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()

        if i % config.policy.learn.log_freq == 0 or not is_train or is_first_epoch:
            metrics = dict()
            metrics['loss'] = loss_mean.item()

            images = get_log_visualization(
                rgb_image, birdview, command, loss, pred_location,
                (_pred_location + 1) * coord_converter._img_size / 2, _teac_location
            )

            log_saver.scalar(is_train=is_train, loss_mean=loss_mean.item())
            log_saver.image(is_train=is_train, birdview=images)

        log_saver.scalar(is_train=is_train, fps=1.0 / (time.time() - tick))

        tick = time.time()

        if is_first_epoch and i == 10:
            iterator_tqdm.close()
            break


def main(cfg):
    if cfg.policy.cudnn:
        torch.backends.cudnn.benchmark = True

    train_dataset = LBCImageDataset(**cfg.data.train)
    val_dataset = LBCImageDataset(**cfg.data.val)

    train_dataloader = DataLoader(
        train_dataset, cfg.policy.learn.batch_size, num_workers=16, shuffle=True, drop_last=True, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, cfg.policy.learn.batch_size, num_workers=16, shuffle=False, drop_last=False, pin_memory=True
    )

    criterion = LocationLoss(**cfg.camera_args)

    coord_converter = CoordConverter(**cfg.camera_args)

    lbc_policy = LBCImagePolicy(cfg.policy)

    teacher_policy = LBCBirdviewPolicy(cfg.teacher_policy)
    teacher_policy.learn_mode.load_state_dict(torch.load(cfg.teacher_policy.ckpt_path))

    _epoch = 0

    if cfg.policy.model.resume:
        log_dir = Path('./log/{}/'.format(cfg.exp_name))
        checkpoints = list(log_dir.glob('model-*.th'))
        checkpoint = str(checkpoints[-1])
        _epoch = int(checkpoint[24:-3])
        print(f'loading {checkpoint}')
        lbc_policy.learn_mode.load_state_dict(torch.load(checkpoint))

    log_saver = Experiment(log_dir='./log/{}/'.format(cfg.exp_name))
    optim = torch.optim.Adam(lbc_policy._model.parameters(), lr=cfg.policy.learn.lr)

    for epoch in tqdm(range(_epoch + 1, cfg.policy.learn.epoches + 1), desc='Epoch'):
        train_or_eval(
            coord_converter, criterion, lbc_policy.learn_mode, teacher_policy.learn_mode, train_dataloader, optim, True,
            cfg, epoch == 0, log_saver
        )
        train_or_eval(
            coord_converter, criterion, lbc_policy.learn_mode, teacher_policy.learn_mode, val_dataloader, optim, False,
            cfg, epoch == 0, log_saver
        )

        if epoch in [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1000]:
            torch.save(lbc_policy.learn_mode.state_dict(), './log/{}/model-{}.th'.format(cfg.exp_name, epoch))

        log_saver.end_epoch()


if __name__ == '__main__':
    main(main_config)
