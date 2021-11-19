import os
import numpy as np
from functools import partial
from easydict import EasyDict
import copy
import time
from tensorboardX import SummaryWriter

from core.envs import SimpleCarlaEnv
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SerialEvaluator
from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import PPOPolicy
from ding.worker import BaseLearner, SampleSerialCollector
from ding.utils import set_pkg_seed

from demo.simple_rl.model import PPORLModel
from demo.simple_rl.env_wrapper import ContinuousBenchmarkEnvWrapper
from core.utils.data_utils.bev_utils import unpack_birdview
from core.utils.others.ding_utils import compile_config

train_config = dict(
    exp_name='ppo21_bev32_lr1e4_bs128_ns3000_update5_train_ft',
    env=dict(
        collector_env_num=7,
        evaluator_env_num=1,
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
            ),
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        ignore_light=True,
        replay_path='./ppo_video',
        visualize=dict(
            type='birdview',
        ),
        manager=dict(
            collect=dict(
                auto_reset=True,
                shared_memory=False,
                context='spawn',
                max_retry=1,
            ),
            eval=dict()
        ),
        wrapper=dict(
            # Collect and eval suites for training
            collect=dict(suite='train_ft', ),
            eval=dict(suite='FullTown02-v1', ),
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9016, 2]),
    ],
    policy=dict(
        cuda=True,
        nstep_return=False,
        on_policy=True,
        model=dict(
            action_shape=2,
        ),
        learn=dict(
            epoch_per_collect=5,
            batch_size=128,
            learning_rate=0.0001,
            weight_decay=0.0001,
            value_weight=0.5,
            adv_norm=False,
            entropy_weight=0.01,
            clip_ratio=0.2,
            target_update_freq=100,
            learner=dict(
                hook=dict(
                    log_show_after_iter=1000,
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
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=5000,
                n_episode=5,
                stop_rate=0.7,
                transform_obs=True,
            ),
        ),
    ),
)

main_config = EasyDict(train_config)


def wrapped_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    return ContinuousBenchmarkEnvWrapper(SimpleCarlaEnv(env_cfg, host, port, tm_port), wrapper_cfg)


def main(cfg, seed=0):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
    )
    tcp_list = parse_carla_tcp(cfg.server)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    assert len(tcp_list) >= collector_env_num + evaluator_env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(
            collector_env_num + evaluator_env_num, len(tcp_list)
    )

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.collect, *tcp_list[i]) for i in range(collector_env_num)],
        cfg=cfg.env.manager.collect,
    )
    evaluate_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.eval, *tcp_list[collector_env_num + i]) for i in range(evaluator_env_num)],
        cfg=cfg.env.manager.eval,
    )
    # Uncomment this to add save replay when evaluation
    # evaluate_env.enable_save_replay(cfg.env.replay_path)
    collector_env.seed(seed)
    evaluate_env.seed(seed)
    set_pkg_seed(seed)

    model = PPORLModel(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = SerialEvaluator(cfg.policy.eval.evaluator, evaluate_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name)

    learner.call_hook('before_run')

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Sampling data from environments
        new_data = collector.collect(n_sample=3000, train_iter=learner.train_iter)
        unpack_birdview(new_data)
        learner.train(new_data, collector.envstep)
    learner.call_hook('after_run')

    collector_env.close()
    evaluate_env.close()
    evaluator.close()
    learner.close()


if __name__ == '__main__':
    main(main_config)
