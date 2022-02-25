import metadrive
import gym
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import BaseEnvManager, SyncSubprocessEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy, DQNPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, AdvancedReplayBuffer
from core.envs import DriveEnvWrapper, MetaDriveMacroEnv

metadrive_macro_config = dict(
    exp_name='metadrive_macro_dqn_eval',
    env=dict(
        metadrive=dict(use_render=True),
        manager=dict(
            shared_memory=False,
            max_retry=2,
            context='spawn',
        ),
        n_evaluator_episode=2,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
        wrapper=dict(),
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=5,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
    ),
)

main_config = EasyDict(metadrive_macro_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveMacroEnv(env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        save_cfg=True
    )
    print(cfg.policy.collect.collector)

    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    policy = DQNPolicy(cfg.policy)
    import torch
    state_dict = torch.load('iteration_40000.pth.tar', map_location='cpu')
    policy.eval_mode.load_state_dict(state_dict)

    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )

    for i in range(5):
        print('total {} times'.format(i))
        stop, rate = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    main(main_config)
