import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from easydict import EasyDict
from tqdm import tqdm

from core.models import VanillaVAE
from core.data import BeVVAEDataset
from core.utils.simulator_utils.carla_utils import visualize_birdview


config = dict(
    exp_name='vae_naive_train',
    data=dict(
        train=dict(
            root_dir='naive_bev_train',
        ),
        val=dict(
            root_dir='naive_bev_val',
        ),
    ),
    learn=dict(
        batch_size=128,
        lr=1e-4,
        weight_decay=0.999,
        epoches=100,
        val_freq=5,
    ),
    model=dict(
        in_channels=7,
        latent_dim=128,
    ),
)
main_config = EasyDict(config)


def _preprocess_image(x):
    """
    Takes -
    list of (h, w, 3)
    tensor of (n, h, 3)
    """
    ret = []
    for b in range(x.shape[0]):
        bev = x[b, ...].squeeze()
        bev = bev.detach().cpu().numpy().transpose(1, 2, 0)
        bev = visualize_birdview(bev)
        bev = torch.Tensor(bev.transpose(2, 0, 1))
        ret.append(bev)
    x = torch.stack(ret)
    #x = torch.nn.functional.interpolate(x, 128, mode='nearest')
    x = vutils.make_grid(ret, padding=2, normalize=True, nrow=4)
    x = x.cpu().numpy()

    return x


def main(cfg):
    train_dataset = BeVVAEDataset(**cfg.data.train)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.learn.batch_size, num_workers=12, pin_memory=True)
    val_dataset = BeVVAEDataset(**cfg.data.val)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.learn.batch_size, num_workers=12, pin_memory=True, drop_last=True, shuffle=True)

    model = VanillaVAE(**cfg.model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learn.lr, weight_decay=cfg.learn.weight_decay)
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))

    iter_num = 0
    for epoch in range(cfg.learn.epoches):
        model.train()
        for data in tqdm(train_dataloader, desc='Train'):
            bev = data['birdview'].cuda()
            ret = model.forward(bev)
            ret = model.loss_function(*ret)
            loss = ret['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_num % 50 == 0:
                for k, v in ret.items():
                    tb_logger.add_scalar("train_iter/{}".format(k), v.item(), iter_num)
            iter_num += 1

        if epoch > 0 and epoch % cfg.learn.val_freq == 0:
            model.eval()
            total_loss = {}
            for data in tqdm(val_dataloader, desc='Val'):
                with torch.no_grad():
                    bev = data['birdview'].cuda()
                    ret = model.forward(bev)
                    loss = model.loss_function(*ret)
                    for k, v in loss.items():
                        if k not in total_loss:
                            total_loss[k] = [v]
                        else:
                            total_loss[k].append(v)
            total_loss_mean = {k: torch.stack(v).mean().item() for k, v in total_loss.items()}
            for k, v in total_loss_mean.items():
                tb_logger.add_scalar("val_epoch/{}_avg".format(k), v, epoch)

            test_sample = next(iter(val_dataloader))['birdview']
            test_sample = test_sample[:16, ...].cuda()
            with torch.no_grad():
                recon_sample = model.generate(test_sample, current_device='cuda')
                random_sample = model.sample(16, current_device='cuda')
            tb_logger.add_image('rec_bev', _preprocess_image(recon_sample), epoch)
            tb_logger.add_image('ran_bev', _preprocess_image(random_sample), epoch)
            if not os.path.exists('./ckpt'):
                os.makedirs('./ckpt')
            state_dict = model.state_dict()
            torch.save(state_dict, "./ckpt/{}_{}_ckpt".format(cfg.exp_name, epoch))


if __name__ == '__main__':
    main(main_config)
