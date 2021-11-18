import copy
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict
from loguru import logger
import torch
import torchvision
from tensorboardX import SummaryWriter


def _preprocess_image(x):
    """
    Takes -
    list of (h, w, 3)
    tensor of (n, h, 3)
    """
    if isinstance(x, list):
        x = np.stack(x, 0).transpose(0, 3, 1, 2)

    x = torch.Tensor(x)

    if x.requires_grad:
        x = x.detach()

    if x.dim() == 3:
        x = x.unsqueeze(1)

    # x = torch.nn.functional.interpolate(x, 128, mode='nearest')
    x = torchvision.utils.make_grid(x, padding=2, normalize=True, nrow=4)
    x = x.cpu().numpy()

    return x


def _format(**kwargs):
    result = list()

    for k, v in kwargs.items():
        if isinstance(v, float) or isinstance(v, np.float32):
            result.append('%s: %.2f' % (k, v))
        else:
            result.append('%s: %s' % (k, v))

    return '\t'.join(result)


class Experiment(object):

    def __init__(self, log_dir):
        """
        This MUST be called.
        """
        self._log = logger
        self.epoch = 0
        self.scalars = OrderedDict()

        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        for i in self._log._handlers:
            self._log.remove(i)

        self._writer_train = SummaryWriter(str(self.log_dir / 'train'))
        self._writer_val = SummaryWriter(str(self.log_dir / 'val'))
        self._log.add(str(self.log_dir / 'log.txt'), format='{time:MM/DD/YY HH:mm:ss} {level}\t{message}')

        # Functions.
        self.debug = self._log.debug
        self.info = lambda **kwargs: self._log.info(_format(**kwargs))

    def load_config(self, model_path):
        log_dir = Path(model_path).parent

        with open(str(log_dir / 'config.json'), 'r') as f:
            return json.load(f)

    def save_config(self, config_dict):

        def _process(x):
            for key, val in x.items():
                if isinstance(val, dict):
                    _process(val)
                elif not isinstance(val, float) and not isinstance(val, int):
                    x[key] = str(val)

        config = copy.deepcopy(config_dict)

        _process(config)

        with open(str(self.log_dir / 'config.json'), 'w+') as f:
            json.dump(config, f, indent=4, sort_keys=True)

    def scalar(self, is_train=True, **kwargs):
        for k, v in sorted(kwargs.items()):
            key = (is_train, k)

            if key not in self.scalars:
                self.scalars[key] = list()

            self.scalars[key].append(v)

    def image(self, is_train=True, **kwargs):
        writer = self._writer_train if is_train else self._writer_val

        for k, v in sorted(kwargs.items()):
            writer.add_image(k, _preprocess_image(v), self.epoch)

    def end_epoch(self, net=None):
        for (is_train, k), v in self.scalars.items():
            info = OrderedDict()
            info['%s_%s' % ('train' if is_train else 'val', k)] = np.mean(v)
            info['std'] = np.std(v, dtype=np.float32)
            info['min'] = np.min(v)
            info['max'] = np.max(v)
            info['n'] = len(v)

            self.info(**info)

            if is_train:
                self._writer_train.add_scalar(k, np.mean(v), self.epoch)
            else:
                self._writer_val.add_scalar(k, np.mean(v), self.epoch)

        self.scalars.clear()

        if net is not None:
            if self.epoch % 10 == 0:
                torch.save(net.state_dict(), str(self.log_dir / ('model_%03d.t7' % self.epoch)))

            torch.save(net.state_dict(), str(self.log_dir / 'latest.t7'))

        self.epoch += 1
