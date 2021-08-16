from easydict import EasyDict
import torch
from functools import partial

from core.envs import SimpleCarlaEnv
from core.policy import LBCImagePolicy
from core.eval import CarlaBenchmarkEvaluator
from core.utils.others.tcp_helper import parse_carla_tcp
from core.utils.others.config_helper import deep_merge_dicts
from ding.envs import AsyncSubprocessEnvManager
from ding.utils import set_pkg_seed
from demo.lbc.lbc_env_wrapper import LBCEnvWrapper


lbc_config = dict(
    env_num=5,
    env=dict(
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            n_vehicles=10,
            n_pedestrians=10,
            verbose=False,
            planner=dict(
                type='lbc',
                resolution=2.5,
                threshold_before=9.0,
                threshold_after=1.5,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[384, 160],
                    position=[2.0, 0.0, 1.4],
                ),
            ),
        ),
    ),
    env_manager=dict(
        shared_memory=False,
        auto_reset=False,
    ),
    env_wrapper=dict(),
    server=[dict(carla_host='localhost', carla_ports=[9000, 9002, 2])],
    policy=dict(
        ckpt_path='model-20.th',
    ),
    eval=dict(
        suite='FullTown01-v3',
        episodes_per_suite=50,
    )
)

main_config = EasyDict(lbc_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return LBCEnvWrapper(SimpleCarlaEnv(env_cfg, host, port))


def main(cfg, seed=0):
    cfg.env_manager = deep_merge_dicts(AsyncSubprocessEnvManager.default_config(), cfg.env_manager)
    tcp_list = parse_carla_tcp(cfg.server)

    carla_env = AsyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, *tcp_list[i]) for i in range(cfg.env_num)],
        cfg=cfg.env_manager,
    )

    carla_env.seed(seed)
    set_pkg_seed(seed)
    lbc_policy = LBCImagePolicy(cfg.policy).eval_mode
    state_dict = torch.load(cfg.policy.ckpt_path)
    lbc_policy.load_state_dict(state_dict)

    evaluator = CarlaBenchmarkEvaluator(cfg.eval, carla_env, lbc_policy)
    evaluator.eval()

    evaluator.close()


if __name__ == '__main__':
    main(main_config)
