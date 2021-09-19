'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

from functools import partial
import numpy as np
import torch
import PIL.Image as Image
import os

from easydict import EasyDict
from ding.envs import SyncSubprocessEnvManager
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.eval import CarlaBenchmarkEvaluator
from core.utils.others.tcp_helper import parse_carla_tcp
from ding.torch_utils.data_helper import to_tensor
from demo.cict_demo.cict_policy import CICTPolicy

eval_config = dict(
    env=dict(
        env_num=5,
        simulator=dict(
            verbose=False,
            disable_two_wheels=True,
            waypoint_num=32,
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[640, 360],
                    position=[0.5, 0.0, 2.5],
                    rotation=[0, 0, 0],
                ),
                dict(
                    name='lidar',
                    type='lidar',
                    channels=64,
                    range=50,
                    points_per_second=100000,
                    rotation_frequency=30,
                    upper_fov=10,
                    lower_fov=-30,
                    position=[0.5, 0.0, 2.5],
                    rotation=[0, 0, 0],
                )
            ),
            planner=dict(type='behavior', resolution=1),
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        manager=dict(
            shared_memory=False,
            auto_reset=False,
            context='spawn',
            max_retry=1,
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9010, 2])
    ],
    policy=dict(
        target_speed=25,
        eval=dict(
            evaluator=dict(
                suite='FullTown01-v1',
                episodes_per_suite=5,
                transform_obs=True,
            ),
        ),
    ),
)

policy_config = dict(
    model=dict(
        gan_ckpt_path='_logs/sample/cict_GAN/checkpoints/3000.pth',
        traj_ckpt_path='_logs/sample/cict_traj/checkpoints/65000.pth'
    ),
    SAVE_DIR='vis',
    IMG_HEIGHT=128,
    IMG_WIDTH=256,
    SENSORS=dict(rgb=[3, 360, 640]),
    DEST=0,
    SPEED_FACTOR=25.0,
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
        traj_model=dict(input_dim=1, hidden_dim=256, out_dim=2)
    ),
    MAX_DIST=25.,
    MAX_T=1,
    IMG_STEP=1,
    PRED_LEN=10,
    DT=0.1,
    PRED_T=3
)
main_config = EasyDict(eval_config)

main_config.policy.update(policy_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return CarlaEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port))


'''
def dataset_test():
    data_dir = '/data3/yg/cict_datasets_train_d'
    _policy = CICTPolicy(main_config.policy).eval_mode

    for n in range(10):
        npy_path = '_preloads2/episode_%05d.npy' % n
        episode_path = 'episode_%05d' % n
        img_name, _, dest_name2, _, _, measurement = np.load(npy_path, allow_pickle=True)
        _policy.reset([n])
        for i in range(220, len(dest_name2)):
            data = {}
            img = Image.open(os.path.join(data_dir, img_name[i])).convert("RGB")
            data['rgb'] = np.array(img)[:, :, ::-1].copy()
            #print(data['rgb'].shape)
            data['timestamp'] = measurement[i]['time']
            data['location'] = measurement[i]['location']
            data['rotation'] = measurement[i]['rotation']
            data['velocity'] = measurement[i]['velocity']
            data['lidar'] = np.load(os.path.join(data_dir, img_name[i].replace('rgb','lidar').replace('png', 'npy')))
            data['waypoint_list'] = np.load(
                os.path.join(data_dir, img_name[i].replace('rgb','waypoints').replace('png', 'npy')))
            real_control = {}
            real_control['steer'] = measurement[i]['steer']
            real_control['brake'] = measurement[i]['brake']
            real_control['throttle'] = measurement[i]['throttle']
            data = to_tensor(data)
            pred_control = _policy.forward({'0': data})
            pred_control = pred_control['0']
            print(n, i)
            print(pred_control)
            print(real_control)
'''


def main(cfg, seed=0):
    cfg.env.manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env.manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num
    assert len(tcp_list) >= env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(env_num, len(tcp_list))

    evaluate_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )
    evaluate_env.seed(seed)
    set_pkg_seed(seed)
    policy = CICTPolicy(cfg.policy)
    evaluator = CarlaBenchmarkEvaluator(cfg.policy.eval.evaluator, evaluate_env, policy.eval_mode)
    evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
    #dataset_test()
