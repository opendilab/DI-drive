import os
import torch
import cv2
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

from core.policy import LBCBirdviewPolicy, LBCImagePolicy
from core.data import LBCImageDataset
from core.utils.simulator_utils.carla_utils import visualize_birdview
from core.utils.learner_utils.log_saver_utils import Experiment

lbc_config = dict(
    exp_name='lbc_img_train_p0',
    data=dict(
        train=dict(
            root_dir='lbc_datasets_train',
            gap=5,
            n_step=5,
        ),
        val=dict(
            root_dir='lbc_datasets_val',
            gap=5,
            n_step=5,
        ),
    ),
    policy=dict(
        cudnn=True,
        cuda=True,
        model=dict(
            backbone='resnet34',
            pretrained=True,
            all_branch=False,
        ),
        learn=dict(
            epoches=2,
            log_freq=1000,
            batch_size=128,
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
        )
    ),
    teacher_policy=dict(ckpt_path='model-256.th', ),
)
main_config = EasyDict(lbc_config)

CROP_SIZE = main_config.policy.camera_args.crop_size


class CoordConverter():

    def __init__(
        self,
        w=384,
        h=160,
        fov=90,
        world_y=1.4,
        fixed_offset=4.0,
        n_step=5,
        crop_size=192,
        pixels_per_meter=5,
        device='cuda'
    ):
        self._w = w
        self._h = h
        self._img_size = torch.FloatTensor([w, h]).to(device)
        self._fov = fov
        self._world_y = world_y
        self._fixed_offset = fixed_offset
        self._n_step = n_step
        self._crop_size = crop_size
        self._pixels_per_meter = pixels_per_meter

        self._tran = np.array([0., 0., 0.])
        self._rot = np.array([0., 0., 0.])
        f = self._w / (2 * np.tan(self._fov * np.pi / 360))
        self._A = np.array([[f, 0., self._w / 2], [0, f, self._h / 2], [0., 0., 1.]])

    def _project_image_xy(self, xy):
        N = len(xy)
        xyz = np.zeros((N, 3))
        xyz[:, 0] = xy[:, 0]
        xyz[:, 1] = 1.4
        xyz[:, 2] = xy[:, 1]

        image_xy, _ = cv2.projectPoints(xyz, self._tran, self._rot, self._A, None)
        image_xy[..., 0] = np.clip(image_xy[..., 0], 0, self._w)
        image_xy[..., 1] = np.clip(image_xy[..., 1], 0, self._h)

        return image_xy[:, 0]

    def __call__(self, map_locations):
        teacher_locations = map_locations.detach().cpu().numpy()
        teacher_locations = (teacher_locations + 1) * self._crop_size / 2
        N = teacher_locations.shape[0]
        teacher_locations[:, :, 1] = self._crop_size - teacher_locations[:, :, 1]
        teacher_locations[:, :, 0] -= self._crop_size / 2
        teacher_locations = teacher_locations / self._pixels_per_meter
        teacher_locations[:, :, 1] += self._fixed_offset
        teacher_locations = self._project_image_xy(np.reshape(teacher_locations, (N * self._n_step, 2)))
        teacher_locations = np.reshape(teacher_locations, (N, self._n_step, 2))
        teacher_locations = torch.FloatTensor(teacher_locations)

        return teacher_locations


class LocationLoss(torch.nn.Module):

    def __init__(self, w=384, h=160, device='cuda', **kwargs):
        super().__init__()
        self._img_size = torch.FloatTensor([w, h]).to(device)

    def forward(self, pred_locations, locations):
        locations = locations.to(pred_locations.device)
        locations = locations / (0.5 * self._img_size) - 1
        return torch.mean(torch.abs(pred_locations - locations), dim=(1, 2))


def get_log_visualization(rgb_image, birdview, command, loss, pred_locations, teac_locations, _teac_locations, size=32):
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
        for x, y in teac_locations[i]:
            _dot(rgb, x, y, BLUE)
        for x, y in pred_locations[i]:
            _dot(rgb, x, y, RED)

        _write('Command: %s' % _command, 1, 0)
        _write('Loss: %.2f' % loss[i].item(), 2, 0)

        images.append((loss[i].item(), _stick_together(rgb, canvas)))

    return [x[1] for x in sorted(images, reverse=True, key=lambda x: x[0])]


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

    for i, data in enumerate(iterator_tqdm):
        rgb_data = {
            'rgb': data['rgb'],
            'command': data['command'],
            'location': data['location'],
            'speed': data['speed']
        }
        birdview_data = {
            'birdview': data['birdview'],
            'command': data['command'],
            'location': data['location'],
            'speed': data['speed']
        }
        with torch.no_grad():
            teacher_res_dict = teacher_policy.forward(birdview_data)
        res_dict = policy.forward(rgb_data)

        _pred_location = res_dict['locations_pred']
        _teac_loaction = teacher_res_dict['locations_pred']
        pred_location = (_pred_location + 1) * coord_converter._img_size / 2
        teac_location = coord_converter(_teac_loaction)

        loss = criterion(_pred_location, teac_location)
        loss_mean = loss.mean()

        if is_train and not is_first_epoch:
            optim.zero_grad()
            loss_mean.backward()
            optim.step()

        if i % config.policy.learn.log_freq == 0 or not is_train or is_first_epoch:
            metrics = dict()
            metrics['loss'] = loss_mean.item()

            images = get_log_visualization(
                data['rgb'], data['birdview'], data['command'], loss, pred_location, teac_location,
                _teac_loaction
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

    criterion = LocationLoss(**cfg.policy.camera_args)

    coord_converter = CoordConverter(**cfg.policy.camera_args)

    lbc_policy = LBCImagePolicy(cfg.policy)

    teacher_policy = LBCBirdviewPolicy(cfg.teacher_policy)
    teacher_policy.learn_mode.load_state_dict(torch.load(cfg.teacher_policy.ckpt_path))

    log_saver = Experiment(log_dir='./log/{}/'.format(cfg.exp_name))
    optim = torch.optim.Adam(lbc_policy._model.parameters(), lr=cfg.policy.learn.lr)

    for epoch in tqdm(range(cfg.policy.learn.epoches + 1), desc='Epoch'):
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
