from easydict import EasyDict
import torch

from core.envs import SimpleCarlaEnv
from core.policy import CILRSPolicy
from core.eval import SingleCarlaEvaluator
from core.utils.others.tcp_helper import parse_carla_tcp
from ding.utils import set_pkg_seed
from demo.cilrs.cilrs_env_wrapper import CILRSEnvWrapper

cilrs_config = dict(
    env=dict(
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
        visualize=dict(
            type='rgb',
            outputs=['show']
        ),
        wrapper=dict(),
        col_is_failure=True,
        stuck_is_failure=True,
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
                #render=True,
                transform_obs=True,
            ),
        )
    ),
)

main_config = EasyDict(cilrs_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return CILRSEnvWrapper(SimpleCarlaEnv(env_cfg, host, port))


def main(cfg, seed=0):
    tcp_list = parse_carla_tcp(cfg.server)
    assert len(tcp_list) > 0, "No Carla server found!"

    carla_env = wrapped_env(cfg.env, *tcp_list[0])
    carla_env.seed(seed)
    set_pkg_seed(seed)
    cilrs_policy = CILRSPolicy(cfg.policy).eval_mode
    if cfg.policy.ckpt_path is not None:
        state_dict = torch.load(cfg.policy.ckpt_path)
        cilrs_policy.load_state_dict(state_dict)

    evaluator = SingleCarlaEvaluator(cfg.policy.eval.evaluator, carla_env, cilrs_policy)
    res = evaluator.eval()

    evaluator.close()


if __name__ == '__main__':
    main(main_config)
