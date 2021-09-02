'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

from easydict import EasyDict
from ding.utils import set_pkg_seed

from core.envs import SimpleCarlaEnv, CarlaEnvWrapper
from core.eval import SingleCarlaEvaluator
from core.policy import AutoPIDPolicy
from core.utils.others.tcp_helper import parse_carla_tcp

autorun_config = dict(
    env=dict(
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            n_vehicles=10,
            n_pedestrians=10,
            verbose=False,
            planner=dict(type='basic', ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[800, 600],
                    position=[-5.5, 0, 2.8],
                    rotation=[-15, 0, 0],
                ),
                dict(
                    name='birdview',
                    type='bev',
                    size=[500, 500],
                    pixels_per_meter=8,
                ),
            ),
        ),
        visualize=dict(
            type='birdview',
            outputs=['show']
        ),
    ),
    env_wrapper=dict(),
    server=[dict(carla_host='localhost', carla_ports=[9000, 9002, 2])],
    eval=dict(
        render=True,
        reset_param=dict(
            start=0,
            end=2,
        ),
    ),
    policy=dict(target_speed=40, ),
)

main_config = EasyDict(autorun_config)


def main(cfg, seed=0):
    tcp_list = parse_carla_tcp(cfg.server)
    host, port = tcp_list[0]

    carla_env = CarlaEnvWrapper(SimpleCarlaEnv(cfg.env, host, port), cfg.env_wrapper)
    carla_env.seed(seed)
    set_pkg_seed(seed)
    auto_policy = AutoPIDPolicy(cfg.policy)
    evaluator = SingleCarlaEvaluator(cfg.eval, carla_env, auto_policy.eval_mode)
    evaluator.eval(cfg.eval.reset_param)
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
