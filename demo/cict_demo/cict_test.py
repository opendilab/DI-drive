'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

import numpy as np
import os

from easydict import EasyDict
from ding.utils import set_pkg_seed
from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.eval import SingleCarlaEvaluator
from core.utils.others.tcp_helper import parse_carla_tcp
from demo.cict_demo.cict_policy import CICTPolicy

autoeval_config = dict(
    env=dict(
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
        visualize=dict(type='rgb', outputs=['show']),
        wrapper=dict(
            suite='FullTown02-v1',
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9002, 2])
    ],
    policy=dict(
        eval=dict(
            evaluator=dict(
                render=True,
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
main_config = EasyDict(autoeval_config)

main_config.policy.update(policy_config)


def main(cfg, seed=0):
    tcp_list = parse_carla_tcp(cfg.server)
    assert len(tcp_list) > 0, "No Carla server found!"
    host, port = tcp_list[0]

    carla_env = CarlaEnvWrapper(SimpleCarlaEnv(cfg.env, host, port,), cfg.env.wrapper)
    carla_env.seed(seed)
    set_pkg_seed(seed)
    policy = CICTPolicy(cfg.policy)
    evaluator = SingleCarlaEvaluator(cfg.policy.eval.evaluator, carla_env, policy.eval_mode)
    evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
    #dataset_test()
