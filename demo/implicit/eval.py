import os
import argparse
from easydict import EasyDict
from functools import partial

from core.envs import SimpleCarlaEnv, DriveEnvWrapper
from eval_policy import ImplicitPolicy
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import CarlaBenchmarkEvaluator
from ding.envs import SyncSubprocessEnvManager
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

eval_config = dict(
    env=dict(
        env_num=8,
        simulator=dict(
            verbose=False,
            planner=dict(type='basic', resolution=2.0, min_distance=1.5),
            obs=(
                dict(name='state', type='state'),
                dict(
                    name='rgb',
                    type='rgb',
                    size=[288, 288],
                    fov=100,
                    position=[1.5, 0.0, 2.4],
                    rotation=[0.0, 0.0, 0.0],
                ),
            ),
        ),
        manager=dict(
            shared_memory=False,
            auto_reset=False,
            context='spawn',
            max_retry=1,
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9016, 2]),
    ],
    eval=dict(
        episodes_per_suite=50,
        suite='StraightTown04-v2',
        result_dir='./eval'
    ),
)

main_config = EasyDict(eval_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return DriveEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port))


def main(cfg, policy_cfg, seed=0):
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

    if 'Town04' in cfg.eval.suite or 'town4' in cfg.eval.suite:
        policy_cfg.multi_lanes = True
    else:
        policy_cfg.multi_lanes = False
    policy = ImplicitPolicy(policy_cfg)
    evaluator = CarlaBenchmarkEvaluator(cfg.eval, evaluate_env, policy)
    evaluator.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='base config for DI-drive')
    parser.add_argument(
        "--path-folder-model",
        required=True,
        type=str,
        help="Folder containing all models, ie the supervised Resnet18 and the RL models",
    )
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--nb_action_steering",
        type=int,
        default=27,
        help="How much different steering values in the action (should be odd)",
    )
    parser.add_argument("--max_steering", type=float, default=0.6, help="Max steering value possible in action")
    parser.add_argument(
        "--nb_action_throttle",
        type=int,
        default=3,
        help="How much different throttle values in the action",
    )
    parser.add_argument("--max_throttle", type=float, default=1, help="Max throttle value possible in action")

    parser.add_argument("--front-camera-width", type=int, default=288)
    parser.add_argument("--front-camera-height", type=int, default=288)
    parser.add_argument("--front-camera-fov", type=int, default=100)
    parser.add_argument(
        "--crop-sky",
        action="store_true",
        default=False,
        help="if using CARLA challenge model, let sky, we cropped "
        "it for the models trained only on Town01/train weather",
    )
    args = parser.parse_args()
    main(main_config, args)
