import os
import argparse
import numpy as np
from functools import partial
from easydict import EasyDict
import copy
from tensorboardX import SummaryWriter

from core.envs import SimpleCarlaEnv, BenchmarkEnvWrapper
from core.utils.others.tcp_helper import parse_carla_tcp
from core.eval import SerialEvaluator
from ding.envs import SyncSubprocessEnvManager, BaseEnvManager
from ding.policy import DQNPolicy, PPOPolicy, TD3Policy, SACPolicy, DDPGPolicy
from ding.worker import BaseLearner, SampleSerialCollector, EpisodeSerialCollector, AdvancedReplayBuffer, \
    NaiveReplayBuffer
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from ding.framework import Task
from ding.framework.wrapper import StepTimer

from demo.simple_rl.model import DQNRLModel, PPORLModel, TD3RLModel, SACRLModel, DDPGRLModel
from demo.simple_rl.env_wrapper import DiscreteEnvWrapper, ContinuousEnvWrapper
from core.utils.data_utils.bev_utils import unpack_birdview
from core.utils.others.ding_utils import compile_config
from core.utils.others.ding_utils import read_ding_config


def wrapped_discrete_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    env = SimpleCarlaEnv(env_cfg, host, port, tm_port)
    return BenchmarkEnvWrapper(DiscreteEnvWrapper(env), wrapper_cfg)


def wrapped_continuous_env(env_cfg, wrapper_cfg, host, port, tm_port=None):
    env = SimpleCarlaEnv(env_cfg, host, port, tm_port)
    return BenchmarkEnvWrapper(ContinuousEnvWrapper(env), wrapper_cfg)


def get_cfg(args):
    if args.ding_cfg is not None:
        ding_cfg = args.ding_cfg
    else:
        ding_cfg = {
            'dqn': 'demo.simple_rl.config.dqn_config.py',
            'ppo': 'demo.simple_rl.config.ppo_config.py',
            'td3': 'demo.simple_rl.config.td3_config.py',
            'sac': 'demo.simple_rl.config.sac_config.py',
            'ddpg': 'demo.simple_rl.config.ddpg_config.py',
        }[args.policy]
    default_train_config = read_ding_config(ding_cfg)
    default_train_config.exp_name = args.name
    use_policy, _ = get_cls(args.policy)
    use_buffer = {
        'dqn': AdvancedReplayBuffer,
        'ppo': None,
        'td3': NaiveReplayBuffer,
        'sac': NaiveReplayBuffer,
        'ddpg': NaiveReplayBuffer,
    }[args.policy]
    cfg = compile_config(
        cfg=default_train_config,
        env_manager=SyncSubprocessEnvManager,
        policy=use_policy,
        learner=BaseLearner,
        collector=SampleSerialCollector,
        buffer=use_buffer,
    )
    return cfg


def get_cls(spec):
    policy_cls, model_cls = {
        'dqn': (DQNPolicy, DQNRLModel),
        'ddpg': (DDPGPolicy, DDPGRLModel),
        'td3': (TD3Policy, TD3RLModel),
        'ppo': (PPOPolicy, PPORLModel),
        'sac': (SACPolicy, SACRLModel),
    }[spec]

    return policy_cls, model_cls


def evaluate(task, evaluator, learner):
    def _evaluate(ctx):
        ctx.setdefault("envstep", -1)  # Avoid attribute not existing
        if evaluator.should_eval(learner.train_iter):
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, ctx.envstep)
            if stop:
                task.finish = True
                return
    return _evaluate


def off_policy_collect(epsilon_greedy, collector, replay_buffer, cfg):
    def _collect(ctx):
        ctx.setdefault("train_iter", -1)
        if epsilon_greedy is not None:
            eps = epsilon_greedy(collector.envstep)
            new_data = collector.collect(train_iter=ctx.train_iter, policy_kwargs={'eps': eps})
        else:
            new_data = collector.collect(train_iter=ctx.train_iter)
        ctx.update_per_collect = len(new_data) // cfg.policy.learn.batch_size * 4
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        ctx.envstep = collector.envstep
    return _collect


def on_policy_collect(collector):
    def _collect(ctx):
        ctx.setdefault("train_iter", -1)
        new_data = collector.collect(train_iter=ctx.train_iter)
        unpack_birdview(new_data)
        ctx.new_data = new_data
        ctx.envstep = collector.envstep
    return _collect


def train(learner, replay_buffer, cfg):
    def _train(ctx):
        ctx.setdefault("envstep", -1)
        if 'new_data' in ctx:
            learner.train(ctx.new_data, ctx.envstep)
        else:
            if 'update_per_collect' in ctx:
                update_per_collect = ctx.update_per_collect
            else:
                update_per_collect = cfg.policy.learn.update_per_collect
            for i in range(update_per_collect):
                train_data = replay_buffer.sample(cfg.policy.learn.batch_size, learner.train_iter)
                if train_data is not None:
                    train_data = copy.deepcopy(train_data)
                    unpack_birdview(train_data)
                    learner.train(train_data, ctx.envstep)
                if cfg.policy.get('priority', False):
                    replay_buffer.update(learner.priority_info)
        ctx.train_iter = learner.train_iter
    return _train


def main(args, seed=0):
    cfg = get_cfg(args)
    tcp_list = parse_carla_tcp(cfg.server)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    assert len(tcp_list) >= collector_env_num + evaluator_env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(
            collector_env_num + evaluator_env_num, len(tcp_list)
    )

    if args.policy == 'dqn':
        wrapped_env = wrapped_discrete_env
    else:
        wrapped_env = wrapped_continuous_env

    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, cfg.env.wrapper.collect, *tcp_list[i]) for i in range(collector_env_num)],
        cfg=cfg.env.manager.collect,
    )
    evaluate_env = SyncSubprocessEnvManager(
        env_fn=[
            partial(wrapped_env, cfg.env, cfg.env.wrapper.eval, *tcp_list[collector_env_num + i])
            for i in range(evaluator_env_num)
        ],
        cfg=cfg.env.manager.eval,
    )
    # Uncomment this to add save replay when evaluation
    # evaluate_env.enable_save_replay(cfg.env.replay_path)

    collector_env.seed(seed)
    evaluate_env.seed(seed)
    set_pkg_seed(seed)

    policy_cls, model_cls = get_cls(args.policy)
    model = model_cls(**cfg.policy.model)
    policy = policy_cls(cfg.policy, model=model)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = SerialEvaluator(
        cfg.policy.eval.evaluator, evaluate_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    if args.policy != 'ppo':
        collector = SampleSerialCollector(
            cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
        )
        if cfg.policy.get('priority', False):
            replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
        else:
            replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    else:
        collector = EpisodeSerialCollector(
            cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
        )
        replay_buffer = None

    if args.policy == 'dqn':
        eps_cfg = cfg.policy.other.eps
        epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)
    else:
        epsilon_greedy = None

    learner.call_hook('before_run')

    if cfg.policy.get('random_collect_size', 0) > 0:
        if epsilon_greedy is not None:
            eps = epsilon_greedy(collector.envstep)
            new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs={'eps': eps})
        else:
            new_data = collector.collect(n_sample=cfg.policy.random_collect_size)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)

    with Task(async_mode=args.use_async) as task:
        task.use_step_wrapper(StepTimer(print_per_step=1))
        task.use(evaluate(task, evaluator, learner))
        if replay_buffer is None:
            task.use(on_policy_collect(collector))
        else:
            task.use(off_policy_collect(epsilon_greedy, collector, replay_buffer, cfg))
        task.use(train(learner, replay_buffer, cfg))
        task.run(max_step=int(1e8))

    learner.call_hook('after_run')

    collector.close()
    evaluator.close()
    learner.close()
    if args.policy != 'ppo':
        replay_buffer.close()

    print('finish')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simple-rl train')
    parser.add_argument('-n', '--name', type=str, default='simple-rl', help='experiment name')
    parser.add_argument('-p', '--policy', default='dqn', choices=['dqn', 'ppo', 'td3', 'sac', 'ddpg'], help='RL policy')
    parser.add_argument('-d', '--ding-cfg', default=None, help='DI-engine config path')
    parser.add_argument('--use-async', action='store_true', help='whether use asynchronous execution mode')

    args = parser.parse_args()
    main(args)
