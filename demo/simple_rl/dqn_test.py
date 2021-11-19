import os
import torch
from easydict import EasyDict

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SingleCarlaEvaluator
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from demo.simple_rl.model import DQNRLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper

test_config = dict(
    env=dict(
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            verbose=False,
            waypoint_num=32,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='birdview',
                    type='bev',
                    size=[32, 32],
                    pixels_per_meter=1,
                    pixels_ahead_vehicle=14,
                ),
            )
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        ignore_light=True,
        visualize=dict(type='birdview', outputs=['show']),
        wrapper=dict(
            # Test benchmark suite
            suite='FullTown02-v1',
        ),
    ),
    policy=dict(
        cuda=True,
        # Pre-train model path
        ckpt_path='',
        model=dict(action_shape=21),
        eval=dict(
            evaluator=dict(
                render=True,
                transform_obs=True,
            ),
        ),
    ),
    # Need to change to you own carla server
    server=[dict(
        carla_host='localhost',
        carla_ports=[9000, 9002, 2]
    )],
)

main_config = EasyDict(test_config)


def main(cfg, seed=0):
    cfg.policy = deep_merge_dicts(DQNPolicy.default_config(), cfg.policy)

    tcp_list = parse_carla_tcp(cfg.server)
    assert len(tcp_list) > 0, "No Carla server found!"
    host, port = tcp_list[0]

    carla_env = CarlaEnvWrapper(DiscreteEnvWrapper(SimpleCarlaEnv(cfg.env, host, port)), cfg.env.wrapper)
    carla_env.seed(seed)
    set_pkg_seed(seed)
    model = DQNRLModel(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    if cfg.policy.ckpt_path != '':
        state_dict = torch.load(cfg.policy.ckpt_path, map_location='cpu')
        policy.eval_mode.load_state_dict(state_dict)
    evaluator = SingleCarlaEvaluator(cfg.policy.eval.evaluator, carla_env, policy.eval_mode)
    evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
