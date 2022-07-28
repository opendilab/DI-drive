import torch
from easydict import EasyDict
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import SyncSubprocessEnvManager,BaseEnvManager
from ding.config import compile_config
from ding.policy import PPOPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from core.envs import DriveEnvWrapper
from core.policy.traj_policy.traj_ppo import TrajPPO 
from core.policy.traj_policy.traj_vac import ConvVAC
from core.envs.md_traj_env import MetaDriveTrajEnv 
from core.utils.simulator_utils.md_utils.evaluator_utils import MetadriveEvaluator

metadrive_basic_config = dict(
    exp_name='test_render',
    env=dict(
        metadrive=dict(
            traj_control_mode = 'jerk',
            use_render=True,
            seq_traj_len = 1,
            use_lateral_penalty = False,
            traffic_density = 0.2, 
            use_lateral = True, 
            use_speed_reward = True,
            use_jerk_reward = True,#False
            avg_speed = 6.5,#6.5
            driving_reward = 0.2,
            speed_reward = 0.1,
        ),
        manager=dict(
            shared_memory=False,
            max_retry=5,
            context='spawn',
        ),
        n_evaluator_episode=10,
        stop_value=99999,
        collector_env_num=1,
        evaluator_env_num=1,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            obs_shape=[5, 200, 200],
            action_shape=2,
            action_space='continuous',
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=3e-4,
            learner=dict(
                hook=dict(
                    save_ckpt_after_iter=5000,
                )
            )
        ),
        collect=dict(
            n_sample=300,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=1000,
            ),
        ),
    )
)

main_config = EasyDict(metadrive_basic_config)


def wrapped_env(env_cfg, wrapper_cfg=None):
    return DriveEnvWrapper(MetaDriveTrajEnv(config=env_cfg), wrapper_cfg)


def main(cfg):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        InteractionSerialEvaluator,
    )

    evaluator_env_num = cfg.env.evaluator_env_num
    evaluator_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env.metadrive) for _ in range(evaluator_env_num)],
        cfg=cfg.env.manager,
    )

    model = ConvVAC(**cfg.policy.model)
    policy = TrajPPO(cfg.policy,model=model)
    ckpt_path = 'ckpt_best.pth.tar'
    #policy.eval_mode.load_state_dict(torch.load(ckpt_path))
    tb_logger = SummaryWriter('./log/{}/'.format(cfg.exp_name))
    evaluator = MetadriveEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )


    for _ in range(10):
        stop, rate = evaluator.eval_verbose(None, -1)

    evaluator.close()


if __name__ == '__main__':
    main(main_config)
