import os
import argparse
import copy
from functools import partial
from easydict import EasyDict

import torch
from tensorboardX import SummaryWriter
import numpy as np

from carla_env import ImplicitCarlaEnv
from models import ImplicitDQN
from core.utils.others.ding_utils import compile_config
from core.utils.others.tcp_helper import parse_carla_tcp
from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import DQNPolicy
from ding.worker import BaseLearner, SampleSerialCollector, AdvancedReplayBuffer
from ding.rl_utils import get_epsilon_greedy_fn
from ding.utils import set_pkg_seed
from ding.utils.default_helper import deep_merge_dicts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='carla_dqn.yaml')
    parser.add_argument('--simulator_config_path', type=str, default='carla_rl_simulator.yaml')
    parser.add_argument('--weather_list', type=int, nargs='+', default=[1, 4, 6, 8])
    parser.add_argument('--town', type=str, default='Town04')
    parser.add_argument('--supervised_model_path', type=str, default='models/model_supervised/model_16.pth')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--n_vehicles', type=int, default=0)
    parser.add_argument('--n_pedestrians', type=int, default=0)

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
        default=True,
        help="if using CARLA challenge model, let sky, we cropped "
        "it for the models trained only on Town01/train weather",
    )

    args = parser.parse_known_args()[0]
    return args


nstep = 3
train_config = dict(
    exp_name='implicit_affordance',
    env=dict(
        env_num=8,
        simulator=dict(
            town='Town04',
            disable_two_wheels=True,
            verbose=False,
            planner=dict(
                type='basic',
                resolution=2.5,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[288, 288],
                    fov=100,
                    position=[1.5, 0.0, 2.4],
                    rotation=[0.0, 0.0, 0.0],
                ),
            )
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        manager=dict(
            auto_reset=True,
            shared_memory=False,
            context='spawn',
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9016, 2]),
    ],
    policy=dict(
        cuda=True,
        pirority=True,
        nstep=nstep,
        learn=dict(
            # How many steps to train after collector's one collection. Bigger "train_iteration" means bigger off-policy.
            # collect data -> train fixed steps -> collect data -> ...
            train_iteration=3,
            update_per_collect=100,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=0.0,
            target_update_freq=100,
            max_iterations=1e8,
            learner=dict(
                load_path='',
                hook=dict(
                    save_ckpt_after_iter=dict(
                        name='save_ckpt_after_iter',
                        type='save_ckpt',
                        priority=20,
                        position='after_iter',
                        ext_args=dict(freq=100, ),
                    ),
                    log_show=dict(
                        name='log_show',
                        type='log_show',
                        priority=20,
                        position='after_iter',
                        ext_args=dict(freq=100, ),
                    ),
                ),
            ),
        ),
        eval=dict(
            evaluator=dict(
                render=False,
                final_reward=200,
            ),
        ),
        collect=dict(
            # Cut trajectories into pieces with length "unrol_len".
            episode_num=float('inf'),
            unroll_len=1,
            n_sample=2000,
            collector=dict(),
        ),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(
                replay_buffer_size=80000,
                monitor=dict(print_freq=dict(in_out_count=300, ), ),
            )
        )
    ),
)

main_config = EasyDict(train_config)


def main(cfg, env_args, seed=0):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        buffer=AdvancedReplayBuffer,
    )

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num
    assert len(tcp_list) >= env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(env_num, len(tcp_list))

    env_args.steps_image = [-10, -2, -1, 0]
    env_args.simulator = cfg.env.simulator
    config = env_args
    train_cfgs = []
    for host_port in tcp_list:
        t_config = copy.deepcopy(config)
        t_config.host = host_port[0]
        t_config.port = host_port[1]
        train_cfgs.append(t_config)

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(ImplicitCarlaEnv, train_cfgs[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )
    collector_env.seed(seed)
    set_pkg_seed(seed)

    model = ImplicitDQN(action_space=108, crop_sky=True)
    policy = DQNPolicy(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    learner.call_hook('before_run')
    # Set up other modules, etc. epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    while True:
        if(learner.train_iter > cfg.policy.learn.max_iterations):
            break
        eps = epsilon_greedy(learner.train_iter)
        # Sampling data from environments
        collector_env.reset()
        new_data = collector.collect(n_sample=1000, train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Training
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(cfg.policy.learn.batch_size, learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)

    collector.close()
    learner.close()
    replay_buffer.close()

    learner.call_hook('after_run')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    args = get_args()
    main(main_config, args)
