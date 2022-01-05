import os
import argparse
import torch
from easydict import EasyDict

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SingleCarlaEvaluator
from ding.policy import DQNPolicy, DDPGPolicy, TD3Policy, PPOPolicy, SACPolicy
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from demo.simple_rl.model import DQNRLModel, DDPGRLModel, TD3RLModel, PPORLModel, SACRLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper, ContinuousEnvWrapper


eval_config = dict(
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
        visualize=dict(
            type='birdview',
            #outputs=['show']
        ),
        wrapper=dict(
            suite='FullTown02-v1',
        ),
    ),
    policy=dict(
        cuda=True,
        # Pre-train model path
        ckpt_path='',
        model=dict(),
        eval=dict(
            evaluator=dict(
                render=True,
                transform_obs=True,
            ),
        ),
    ),
    server=[dict(
        carla_host='localhost',
        carla_ports=[9000, 9002, 2]
    )],
)

main_config = EasyDict(eval_config)


def get_cls(spec):
    policy_cls, model_cls = {
        'dqn': (DQNPolicy, DQNRLModel),
        'ddpg': (DDPGPolicy, DDPGRLModel),
        'td3': (TD3Policy, TD3RLModel),
        'ppo': (PPOPolicy, PPORLModel),
        'sac': (SACPolicy, SACRLModel),
    }[spec]

    return policy_cls, model_cls


def main(args, seed=0):
    cfg = main_config
    policy_cls, model_cls = get_cls(args.policy)
    cfg.policy = deep_merge_dicts(policy_cls.default_config(), cfg.policy)

    tcp_list = parse_carla_tcp(cfg.server)
    assert len(tcp_list) > 0, "No Carla server found!"
    host, port = tcp_list[0]

    if args.policy == 'dqn':
        carla_env = CarlaEnvWrapper(DiscreteEnvWrapper(SimpleCarlaEnv(cfg.env, host, port)), cfg.env.wrapper)
    else:
        carla_env = CarlaEnvWrapper(ContinuousEnvWrapper(SimpleCarlaEnv(cfg.env, host, port)), cfg.env.wrapper)
    carla_env.seed(seed)
    set_pkg_seed(seed)

    model = model_cls(**cfg.policy.model)
    policy = policy_cls(cfg.policy, model=model, enable_field=['eval'])

    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
    elif cfg.policy.ckpt_path != '':
        ckpt_path = cfg.policy.ckpt_path
    else:
        ckpt_path = ''
    if ckpt_path != '':
        state_dict = torch.load(ckpt_path, map_location='cpu')
        policy.eval_mode.load_state_dict(state_dict)
    evaluator = SingleCarlaEvaluator(cfg.policy.eval.evaluator, carla_env, policy.eval_mode)
    evaluator.eval()
    evaluator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple-rl test')
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-c', '--ckpt-path', default=None, help='model ckpt path')
    
    args = parser.parse_args()
    main(args)
