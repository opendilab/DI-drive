import os

from torchvision.transforms.transforms import ToTensor
from demo.cict_demo.post import Sensor, params
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision.transforms as transforms
import cv2
from demo.cict_demo.collect_pm import InversePerspectiveMapping
import carla
import numpy as np


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.01)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.01)


class UNetDown(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True, dropout=0.0):
        super(UNetDown, self).__init__()

        norm_layer = nn.InstanceNorm2d if norm else nn.Identity
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), norm_layer(out_channels),
            nn.LeakyReLU(0.2), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm=True, dropout=0.0):
        super(UNetUp, self).__init__()

        norm_layer = nn.InstanceNorm2d if norm else nn.Identity
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            norm_layer(out_channels), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat([x, skip_input], dim=1)
        return x


class GeneratorUNet(nn.Module):

    def __init__(self, params):
        super(GeneratorUNet, self).__init__()

        self.params = params
        self.down_layers = nn.ModuleList()
        #print(self.params)
        for i in range(len(self.params['down_channels']) - 1):
            self.down_layers.append(
                UNetDown(
                    self.params['down_channels'][i],
                    self.params['down_channels'][i + 1],
                    kernel_size=self.params['kernel_size'],
                    stride=self.params['stride'],
                    padding=self.params['padding'],
                    norm=self.params['down_norm'][i],
                    dropout=self.params['down_dropout'][i]
                )
            )

        self.up_layers = nn.ModuleList()
        for i in range(len(self.params['up_channels']) - 1):
            self.up_layers.append(
                UNetUp(
                    self.params['up_channels'][i] + self.params['down_channels'][-i - 1],
                    self.params['up_channels'][i + 1],
                    kernel_size=self.params['kernel_size'],
                    stride=self.params['stride'],
                    padding=self.params['padding'],
                    norm=self.params['up_norm'][i],
                    dropout=self.params['up_dropout'][i]
                )
            )

        self.final_layers = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(
                2 * self.params['up_channels'][-1],
                self.params['final_channels'] * self.params['num_branches'],
                4,
                padding=1
            ), nn.Tanh()
        )

    def forward(self, x, branch):
        d = []
        temp = x
        for down_layer in self.down_layers:
            #print(temp.shape)
            temp = down_layer(temp)
            d.append(temp)

        for i, up_layer in enumerate(self.up_layers):
            temp = up_layer(temp, d[-i - 2])

        output = self.final_layers(temp)
        B, C, H, W = output.shape
        output = output.view(B, self.params['num_branches'], -1, H, W)

        batch_idx = torch.arange(0, B)
        return output[batch_idx.long(), branch.squeeze(1).long()]


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.params = params
        layers = nn.ModuleList()
        for i in range(len(self.params['channels']) - 1):
            layers.append(
                UNetDown(
                    self.params['channels'][i],
                    self.params['channels'][i + 1],
                    kernel_size=self.params['kernel_size'],
                    stride=self.params['stride'],
                    padding=self.params['padding'],
                    norm=self.params['norm'][i],
                    dropout=self.params['dropout'][i]
                )
            )
        self.model = nn.Sequential(
            *layers, nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(self.params['channels'][-1], 1, 4, padding=1, bias=False)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)


class ModelGRU(nn.Module):

    def __init__(self, params):
        super(ModelGRU, self).__init__()
        self.cnn_feature_dim = params['hidden_dim']
        self.rnn_hidden_dim = params['hidden_dim']
        self.cnn = CNN(input_dim=params['input_dim'], out_dim=self.cnn_feature_dim)
        self.gru = nn.GRU(
            input_size=self.cnn_feature_dim,
            hidden_size=self.rnn_hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        self.mlp = MLP_COS(input_dim=self.rnn_hidden_dim + 2, out_dim=params['out_dim'])

    def forward(self, x, t, v0):
        batch_size, timesteps, C, H, W = x.size()

        x = x.view(batch_size * timesteps, C, H, W)
        x = self.cnn(x)

        x = x.view(batch_size, timesteps, -1)
        x, h_n, = self.gru(x)

        x = F.leaky_relu(x[:, -1, :])
        x = self.mlp(x, t, v0)
        return x


class CNN(nn.Module):

    def __init__(self, input_dim=1, out_dim=256):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(input_dim, 64, 5, stride=3, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, self.out_dim, 3, stride=2, padding=1)

        self.bn4 = nn.BatchNorm2d(self.out_dim)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        #x = F.leaky_relu(x)
        x = self.bn4(x)
        x = x.view(-1, self.out_dim)
        return x


class MLP_COS(nn.Module):

    def __init__(self, input_dim=25, out_dim=2):
        super(MLP_COS, self).__init__()
        self.rate = 1.0
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, out_dim)

        self.apply(weights_init)

    def forward(self, x, t, v0):
        B, C = x.shape
        T = t.shape[1]
        x = x.unsqueeze(1).expand(B, T, C)
        v0 = v0.unsqueeze(1).expand(B, T, 1)
        t = t.unsqueeze(-1)
        x = torch.cat([x, t, v0], dim=-1)
        #x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)

        x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(self.rate * x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        return x


class CICTModel():

    def __init__(self, cfg):
        self._cfg = cfg
        self.save_dir = self._cfg.SAVE_DIR
        self.checkpoint_g = torch.load(self._cfg.model.gan_ckpt_path, map_location='cpu')
        self.checkpoint_traj = torch.load(self._cfg.model.traj_ckpt_path, map_location='cpu')

        self._generator = GeneratorUNet(cfg.MODEL_CONFIGURATION['generator'])
        self._model = ModelGRU(cfg.MODEL_CONFIGURATION['traj_model'])
        self._generator.load_state_dict(self.checkpoint_g['state_dict_G'])
        self._model.load_state_dict(self.checkpoint_traj['state_dict'])
        self._generator.eval()
        self._model.eval()

        sensor = Sensor(params.sensor_config['rgb'])
        self._inverse_perspective_mapping = InversePerspectiveMapping(params, sensor)
        self._ipm_buffer = list()

    def clean_buffer(self):
        self._ipm_buffer = list()

    def run_step(self, observation):
        img = observation['rgb'].float()
        dest = observation['dest'].float()
        #print(img.shape, dest.shape)
        branch = torch.LongTensor([0]).unsqueeze(0)
        pm = self._generator(torch.cat([img, dest], dim=1), branch)

        pm = pm.detach().numpy()[0][0] * 127 + 128
        pm = cv2.resize(pm, (self._cfg.SENSORS['rgb'][2], self._cfg.SENSORS['rgb'][1]), interpolation=cv2.INTER_CUBIC)
        lidar = observation['lidar']
        ipm = self._inverse_perspective_mapping.getIPM(pm)
        ipm = self._inverse_perspective_mapping.get_cost_map(ipm, lidar)
        ipm_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        ipm = ipm_transforms(ipm)

        #index = (self._cfg.PRED_T / self._cfg.MAX_T) // self._cfg.DT
        t = torch.arange(0, self._cfg.PRED_LEN).float() * self._cfg.DT + self._cfg.DT
        #t = torch.FloatTensor([self._cfg.PRED_T / self._cfg.MAX_T]).cuda()
        t = t.unsqueeze(0) / self._cfg.MAX_T
        t.requires_grad = True

        cur_v = observation['cur_v'].float()

        while (len(self._ipm_buffer) > 9 * self._cfg.IMG_STEP):
            del self._ipm_buffer[0]
        self._ipm_buffer.append(ipm)

        ipms = []
        for i in range(-9, 1):
            ipms.append(self._ipm_buffer[max(-1 + i * self._cfg.IMG_STEP, -len(self._ipm_buffer))])
        ipms = torch.stack(ipms, dim=0).float()
        T, C, H, W = ipms.shape
        ipms = ipms.unsqueeze(0)
        ipms.requires_grad = True

        pred_xy = self._model(ipms, t, cur_v)

        pred_vx = grad(pred_xy[:, :, 0].sum(), t, create_graph=True)[0] * (self._cfg.MAX_DIST / self._cfg.MAX_T)
        pred_vy = grad(pred_xy[:, :, 1].sum(), t, create_graph=True)[0] * (self._cfg.MAX_DIST / self._cfg.MAX_T)

        pred_vxy = torch.cat([pred_vx.unsqueeze(-1), pred_vy.unsqueeze(-1)], dim=-1)
        #print(pred_vxy)

        pred_ax = grad(pred_vx.sum(), t, create_graph=True)[0] / self._cfg.MAX_T
        pred_ay = grad(pred_vy.sum(), t, create_graph=True)[0] / self._cfg.MAX_T
        #print(pred_ax.shape)
        pred_axy = torch.cat([pred_ax.unsqueeze(-1), pred_ay.unsqueeze(-1)], dim=-1)

        pred_xy = pred_xy * self._cfg.MAX_DIST

        with torch.no_grad():
            theta_a = torch.atan2(pred_ay, pred_ax)
            theta_v = torch.atan2(pred_vy, pred_vx)
            sign = torch.sign(torch.cos(theta_a - theta_v))
            a = torch.mul(torch.norm(pred_axy, dim=-1), sign.flatten()).unsqueeze(1)

            v = torch.norm(pred_vxy, dim=-1)
            k = (pred_vx * pred_ay - pred_vy * pred_ax) / v.unsqueeze(1) ** 3

            trajectory = {
                'xy': pred_xy.data.numpy(),
                'vxy': pred_axy.data.numpy(),
                'axy': pred_axy.data.numpy(),
                'a': a.data.numpy(),
                'theta_v': theta_v.data.numpy(),
                'v': v.data.numpy(),
                'k': k.squeeze(1).cpu().numpy()
            }

        cur_state = {
            'x': 0.,
            'y': 0.,
            'theta': observation['theta'],
            'v': cur_v.data.numpy(),
        }

        target_state = {
            'x': trajectory['xy'][0, self._cfg.PRED_T][0] if trajectory['xy'][0, self._cfg.PRED_T][0] > 0 else 0.,
            'y': trajectory['xy'][0, self._cfg.PRED_T][1],
            'theta': trajectory['theta_v'][0, self._cfg.PRED_T],
            'v': trajectory['v'][0, self._cfg.PRED_T],
            'k': trajectory['k'][0, self._cfg.PRED_T]
        }

        if self.save_dir != '':
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            cv2.imwrite(
                os.path.join(self.save_dir, 'rgb_%s.png' % str(observation['time'])),
                img[0].permute(1, 2, 0).detach().numpy() * 127 + 128
            )
            cv2.imwrite(
                os.path.join(self.save_dir, 'dest_%s.png' % str(observation['time'])),
                dest[0].permute(1, 2, 0).detach().numpy() * 127 + 128
            )
            cv2.imwrite(os.path.join(self.save_dir, 'pm_%s.png' % str(observation['time'])), pm)
            print(
                'time: ' + str(observation['time']) + 'cur_speed:' + str(cur_state['v'][0]) + ' target_speed:' +
                str(target_state['v'])
            )

        return [cur_state, target_state]
