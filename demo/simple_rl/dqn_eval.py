import os
from functools import partial
import torch
from easydict import EasyDict

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import CarlaBenchmarkEvaluator
from ding.envs import AsyncSubprocessEnvManager
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from demo.simple_rl.model import DQNRLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper

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
        model=dict(action_shape=21),
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


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return CarlaEnvWrapper(DiscreteEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg))


def main(cfg, seed=0):
    cfg.policy = deep_merge_dicts(DQNPolicy.default_config(), cfg.policy)
    cfg.env.manager = deep_merge_dicts(AsyncSubprocessEnvManager.default_config(), cfg.env.manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num
    assert len(tcp_list) >= env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(env_num, len(tcp_list))

    carla_env = AsyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )
    carla_env.seed(seed)
    set_pkg_seed(seed)
    model = DQNRLModel(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    if cfg.policy.ckpt_path != '':
        state_dict = torch.load(cfg.policy.ckpt_path, map_location='cpu')
        policy.eval_mode.load_state_dict(state_dict)
    evaluator = CarlaBenchmarkEvaluator(cfg.policy.eval.evaluator, carla_env, policy.eval_mode)
    success_rate = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
