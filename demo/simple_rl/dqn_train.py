import os
import numpy as np
from functools import partial
from easydict import EasyDict
import copy
import time
from tensorboardX import SummaryWriter

from core.envs import SimpleCarlaEnv
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SingleCarlaEvaluator
from ding.envs import SyncSubprocessEnvManager
from ding.policy import DQNPolicy
from ding.worker import BaseLearner, SampleCollector, AdvancedReplayBuffer
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn

from demo.simple_rl.model import DQNRLModel
from demo.simple_rl.env_wrapper import DiscreteBenchmarkEnvWrapper
from core.utils.data_utils.bev_utils import unpack_birdview
from core.utils.others.ding_utils import compile_config

train_config = dict(
    exp_name='dqn21_bev32_buffer400000_lr1e4_train_ft',
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
        stuck_is_failure=False,
        ignore_light=True,
        ran_light_is_failure=False,
        off_road_is_failure=True,
        wrong_direction_is_failure=True,
        off_route_is_failure=True,
        off_route_distance=7.5,
        finish_reward=30,
        #visualize=dict(type='birdview', outputs=['show']),
    ),
    env_num=7,
    env_manager=dict(
        auto_reset=True,
        shared_memory=False,
        context='spawn',
    ),
    env_wrapper=dict(
        train=dict(suite='train_ft', ),
        eval=dict(suite='FullTown02-v1', ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9016, 2]),
    ],
    policy=dict(
        cuda=True,
        priority=True,
        nstep=1,
        learn=dict(
            batch_size=128,
            learning_rate=0.0001,
            weight_decay=0.0001,
            target_update_freq=100,
            learner=dict(
                hook=dict(
                    load_ckpt_before_run='',
                ),
            ),
        ),
        collect=dict(
            collector=dict(
                collect_print_freq=1000,
                deepcopy_obs=True,
                transform_obs=True,
            ),
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(
                replay_buffer_size=400000,
                monitor=dict(
                    sampled_data_attr=dict(
                        print_freq=100,
                    ),
                    periodic_thruput=dict(
                        seconds=120,
                    ),
                ),
            ),
        ),
    ),
    model=dict(
        action_shape=21,
    ),
    eval=dict(
        # render=True,
        eval_freq=5000,
        eval_num=3,
        success_rate=0.7,
        transform_obs=True,
    ),
)

main_config = EasyDict(train_config)


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return DiscreteBenchmarkEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)
    #return MultiDiscreteBenchmarkEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)


def main(cfg, seed=0):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        #DQNMultiDiscretePolicy,
        BaseLearner,
        SampleCollector,
        AdvancedReplayBuffer,
    )
    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env_num

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env_wrapper.train, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env_manager,
    )
    evaluate_env = wrapped_env(cfg.env, cfg.env_wrapper.eval, *tcp_list[env_num])
    collector_env.seed(seed)
    evaluate_env.seed(seed)
    set_pkg_seed(seed)

    model = DQNRLModel(**cfg.model)
    policy = DQNPolicy(cfg.policy, model=model)

    tb_logger = SummaryWriter(os.path.join('./log/', cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = SampleCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = SingleCarlaEvaluator(cfg.eval, evaluate_env, policy.eval_mode)
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger)

    # Set up other modules, etc. epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    learner._instance_name = cfg.exp_name + '_' + time.ctime().replace(' ', '_').replace(':', '_')
    learner.call_hook('before_run')
    eps = epsilon_greedy(learner.train_iter)
    new_data = collector.collect(n_sample=10000, train_iter=learner.train_iter, policy_kwargs={'eps': eps})
    replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)

    while True:
        if evaluator.should_eval(learner.train_iter):
            results_list = []
            for _ in range(cfg.eval.eval_num):
                results_list.append(evaluator.eval())
            success_rate = sum(results_list) / len(results_list)
            if success_rate > cfg.eval.success_rate:
                break
            print("Evaluate success rate: {:.2f}%".format(success_rate*100))
        eps = epsilon_greedy(learner.train_iter)
        # Sampling data from environments
        new_data = collector.collect(n_sample=3000, train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        update_per_collect = len(new_data) // 32
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Training
        for i in range(update_per_collect):
            train_data = replay_buffer.sample(cfg.policy.learn.batch_size, learner.train_iter)
            if train_data is not None:
                train_data = copy.deepcopy(train_data)
                unpack_birdview(train_data)
                learner.train(train_data, collector.envstep)
            replay_buffer.update(learner.priority_info)

    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()
    replay_buffer.close()


if __name__ == '__main__':
    main(main_config)
