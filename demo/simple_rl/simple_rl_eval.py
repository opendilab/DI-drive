import os
import argparse
from functools import partial
import torch
from easydict import EasyDict

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import CarlaBenchmarkEvaluator
from ding.envs import SyncSubprocessEnvManager
from ding.policy import DQNPolicy, DDPGPolicy, TD3Policy, PPOPolicy, SACPolicy
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from demo.simple_rl.model import DQNRLModel, DDPGRLModel, TD3RLModel, PPORLModel, SACRLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper, ContinuousEnvWrapper


eval_config = dict(
    env=dict(
        # Eval env num
        env_num=5,
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
        manager=dict(
            auto_reset=False,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ),
        wrapper=dict(),
    ),
    policy=dict(
        cuda=True,
        # Pre-train model path
        ckpt_path='',
        model=dict(),
        eval=dict(
            evaluator=dict(
                # Eval benchmark suite
                suite='FullTown02-v1',
                episodes_per_suite=25,
                weathers=[1],
                transform_obs=True,
                save_files=True,
            ),
        ),
    ),
    # Need to change to you own carla server
    server=[dict(
        carla_host='localhost',
        carla_ports=[9000, 9010, 2]
    )],
)

main_config = EasyDict(eval_config)


def wrapped_discrete_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    env = SimpleCarlaEnv(env_cfg, host, port, tm_port)
    return CarlaEnvWrapper(DiscreteEnvWrapper(env), wrapper_cfg)


def wrapped_continuous_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    env = SimpleCarlaEnv(env_cfg, host, port, tm_port)
    return CarlaEnvWrapper(ContinuousEnvWrapper(env), wrapper_cfg)


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
    cfg.env.manager = deep_merge_dicts(SyncSubprocessEnvManager.default_config(), cfg.env.manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num
    assert len(tcp_list) >= env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(env_num, len(tcp_list))

    if args.policy == 'dqn':
        wrapped_env = wrapped_discrete_env
    else:
        wrapped_env = wrapped_continuous_env
    carla_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )
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
    evaluator = CarlaBenchmarkEvaluator(cfg.policy.eval.evaluator, carla_env, policy.eval_mode)
    success_rate = evaluator.eval()
    evaluator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple-rl eval')
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-c', '--ckpt-path', default=None, help='model ckpt path')
    
    args = parser.parse_args()
    main(args)
