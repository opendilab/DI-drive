'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

from functools import partial

from easydict import EasyDict
from ding.envs import SyncSubprocessEnvManager
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.eval import CarlaBenchmarkEvaluator
from core.policy import CILPolicy
from core.utils.others.tcp_helper import parse_carla_tcp

autoeval_config = dict(
    env_num=1,
    env=dict(
        simulator=dict(
            verbose=False,
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[800, 600],
                    position=[2.0, 0.0, 1.4],
                    rotation=[-15, 0, 0],
                ), dict(
                    name='birdview',
                    type='bev',
                    size=[500, 500],
                    pixels_per_meter=8,
                )
            ),
            planner=dict(type='behavior', ),
        ),
    ),
    env_manager=dict(
        shared_memory=False,
        auto_reset=False,
    ),
    server=[dict(carla_host='localhost', carla_ports=[5000, 5002, 2])],
    eval=dict(
        suite='FullTown02-v1',
        episodes_per_suite=5,
    ),
    policy=dict(target_speed=40, ),
)

policy_config = dict(
    model=dict(
        ckpt_path='_logs/sample/coil_icra/checkpoints/100.pth'
    ),
    SENSORS=dict(rgb=[3, 88, 200]),
    TARGETS=['steer', 'throttle', 'brake'],
    INPUTS=['speed_module'],
    SPEED_FACTOR=25.0,
    MODEL_TYPE='coil-icra',
    MODEL_CONFIGURATION=dict(
        perception=dict(res=dict(name='resnet34', num_classes=512)),
        measurements=dict(fc=dict(neurons=[128, 128], dropouts=[0.0, 0.0])),
        join=dict(fc=dict(neurons=[512], dropouts=[0.0])),
        speed_branch=dict(fc=dict(neurons=[256, 256], dropouts=[0.0, 0.5])),
        branches=dict(number_of_branches=4, fc=dict(neurons=[256, 256], dropouts=[0.0, 0.5]))
    ),
    PRE_TRAINED=False,
    LEARNING_RATE_DECAY_LEVEL=0.1,
    IMAGE_CUT=[115, 500],
    NUMBER_FRAMES_FUSION=1,
)
main_config = EasyDict(autoeval_config)

main_config.policy.update(policy_config)


def wrapped_env(env_cfg, host, port, tm_port):
    return CarlaEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port))


def main(cfg, seed=0):
    cfg.env_manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env_manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env_num

    evaluate_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, *tcp_list[i], tcp_list[i][1] + 500) for i in range(env_num)],
        cfg=cfg.env_manager,
    )
    evaluate_env.seed(seed)
    set_pkg_seed(seed)
    auto_policy = CILPolicy(cfg.policy)
    evaluator = CarlaBenchmarkEvaluator(cfg.eval, evaluate_env, auto_policy.eval_mode)
    evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
