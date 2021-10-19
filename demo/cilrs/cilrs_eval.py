from easydict import EasyDict
import torch
from functools import partial

from core.envs import SimpleCarlaEnv
from core.policy import CILRSPolicy
from core.eval import CarlaBenchmarkEvaluator
from core.utils.others.tcp_helper import parse_carla_tcp
from ding.utils import set_pkg_seed, deep_merge_dicts
from ding.envs import AsyncSubprocessEnvManager
from demo.cilrs.cilrs_env_wrapper import CILRSEnvWrapper

cilrs_config = dict(
    env=dict(
        env_num=5,
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            verbose=False,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[400, 300],
                    position=[1.3, 0.0, 2.3],
                    fov=100,
                ),
            ),
        ),
        wrapper=dict(),
        col_is_failure=True,
        stuck_is_failure=True,
        manager=dict(
            auto_reset=False,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ),
    ),
    server=[dict(carla_host='localhost', carla_ports=[9000, 9010, 2])],
    policy=dict(
        ckpt_path=None,
        model=dict(
            num_branch=4,
            pretrained=False,
        ),
        eval=dict(
            evaluator=dict(
                suite=['FullTown01-v1'],
                transform_obs=True,
            ),
        )
    ),
)

main_config = EasyDict(cilrs_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return CILRSEnvWrapper(SimpleCarlaEnv(env_cfg, host, port))


def main(cfg, seed=0):
    cfg.env.manager = deep_merge_dicts(AsyncSubprocessEnvManager.default_config(), cfg.env.manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num
    assert len(tcp_list) >= env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(env_num, len(tcp_list))

    carla_env = AsyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )
    carla_env.seed(seed)
    set_pkg_seed(seed)
    cilrs_policy = CILRSPolicy(cfg.policy).eval_mode
    if cfg.policy.ckpt_path is not None:
        state_dict = torch.load(cfg.policy.ckpt_path)
        cilrs_policy.load_state_dict(state_dict)

    evaluator = CarlaBenchmarkEvaluator(cfg.policy.eval.evaluator, carla_env, cilrs_policy)
    success_rate = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
