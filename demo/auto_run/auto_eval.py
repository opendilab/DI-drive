'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

from functools import partial

from easydict import EasyDict
from ding.envs import SyncSubprocessEnvManager
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts

from core.envs import SimpleCarlaEnv, DriveEnvWrapper
from core.eval import CarlaBenchmarkEvaluator
from core.policy import AutoPIDPolicy
from core.utils.others.tcp_helper import parse_carla_tcp

autoeval_config = dict(
    env=dict(
        env_num=4,
        simulator=dict(
            verbose=False,
            obs=(),
            planner=dict(type='behavior', ),
        ),
        manager=dict(
            shared_memory=False,
            auto_reset=False,
            context='spawn',
            max_retry=1,
        ),
    ),
    server=[dict(carla_host='localhost', carla_ports=[9000, 9008, 2])],
    policy=dict(
        target_speed=40,
        eval=dict(
            evaluator=dict(
                suite='FullTown01-v0',
                episodes_per_suite=10,
            ),
        ),
    ),
)

main_config = EasyDict(autoeval_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return DriveEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port))


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
    auto_policy = AutoPIDPolicy(cfg.policy)
    evaluator = CarlaBenchmarkEvaluator(cfg.policy.eval.evaluator, evaluate_env, auto_policy.eval_mode)
    evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
