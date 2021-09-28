import os
import numpy as np
from collections import defaultdict
import torch
from typing import Dict, Any, List, Optional, Callable, Tuple

from .base_evaluator import BaseEvaluator
from core.data.benchmark import ALL_SUITES
from ding.envs import BaseEnvManager
from ding.torch_utils.data_helper import to_tensor
from ding.utils import build_logger, EasyTimer


class SerialEvaluator(BaseEvaluator):
    """
    Evaluator used to serially evaluate a policy for defined times. It is mainly used when training a policy to get the
    evaluator performance frequently and store the best iterations. Different from serial evaluator in `DI-engine`, this
    evaluator compares the performance of iterations by the success rate rather than rewards. You can provide a
    tensorboard logger to save scalars when training.

    Note:
        Env manager must run WITH auto reset.

    :Arguments:
        - cfg (Dict): Config dict.
        - env (BaseEnvManager): Env manager used to evaluate.
        - policy (Any): Policy to evaluate. Must have ``forward`` method.
        - tb_logger (SummaryWriter, optional): Tensorboard writter to store values in tensorboard. Defaults to None.
        - exp_name (str, optional): Name of the experiments. Used to build logger. Defaults to 'default_experiment'.
        - instance_name (str, optional): [description]. Defaults to 'serial_evaluator'.

    :Interfaces: reset, eval, close, should_eval

    :Properties:
        - env (BaseEnvManager): Env manager with several environments used to evaluate.
        - policy (Any): Policy instance to interact with envs.
    """

    config = dict(
        transform_obs=False,
        # Evaluate every "eval_freq" training iterations.
        eval_freq=100,
        n_episode=10,
        stop_rate=0.8,
    )

    def __init__(
            self,
            cfg: Dict,
            env: BaseEnvManager,
            policy: Any,
            tb_logger: Optional['SummaryWriter'] = None,  # noqa
            exp_name: Optional[str] = 'default_experiment',
            instance_name: Optional[str] = 'serial_evaluator',
    ) -> None:
        super().__init__(cfg, env, policy, tb_logger=tb_logger, exp_name=exp_name, instance_name=instance_name)
        self._transform_obs = self._cfg.transform_obs
        self._default_n_episode = self._cfg.n_episode
        self._stop_rate = self._cfg.stop_rate

        self._logger, _ = build_logger(
            path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
        )

        self._last_eval_iter = 0
        self._max_success_rate = 0
        self._timer = EasyTimer()

    @property
    def env(self) -> BaseEnvManager:
        return self._env_manager

    @env.setter
    def env(self, _env_manager: BaseEnvManager) -> None:
        self._end_flag = False
        self._env_manager = _env_manager
        self._env_manager.launch()
        self._env_num = self._env_manager.env_num

    def close(self) -> None:
        """
        Close the collector and the env manager if not closed.
        """
        if self._close_flag:
            return
        self._close_flag = True
        self._env_manager.close()
        if self._tb_logger is not None:
            self._tb_logger.flush()
            self._tb_logger.close()

    def reset(self) -> None:
        """
        Reset evaluator and policies.
        """
        self._policy.reset([i for i in range(self._env_num)])
        self._last_eval_iter = 0
        self._max_success_rate = 0

    def should_eval(self, train_iter: int) -> bool:
        """
        Judge if the training iteration is at frequency value to run evaluation.

        :Arguments:
            - train_iter (int): Current training iteration

        :Returns:
            bool: Whether should run iteration
        """
        if (train_iter - self._last_eval_iter) < self._cfg.eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None
    ) -> Tuple[bool, float]:
        """
        Run evaluation with provided policy arguments. It will evaluate all available episodes of the benchmark suite
        unless `episode_per_suite` is set in config.

        :Arguments:
            - save_ckpt_fn (Callable, optional): Function to save ckpt. Will be called if at best performance.
                Defaults to None.
            - train_iter (int, optional): Current training iterations. Defaults to -1.
            - envstep (int, optional): Current env steps. Defaults to -1.
            - n_episode: (int, optional): Episodes to eval. By default it is set in config.

        :Returns:
            Tuple[bool, float]: Whether reach stop value and success rate.
        """
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        self._env_manager.reset()
        self._policy.reset([i for i in range(self._env_num)])

        episode_count = 0
        results = defaultdict(list)

        with self._timer:
            while episode_count < n_episode:
                obs = self._env_manager.ready_obs
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs)
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                timesteps = self._env_manager.step(actions)
                for env_id, t in timesteps.items():
                    if t.info.get('abnormal', False):
                        self._policy.reset([env_id])
                        continue
                    if t.done:
                        self._policy.reset([env_id])
                        result = {
                            'reward': t.info['final_eval_reward'],
                            'success': t.info['success'],
                            'step': int(t.info['tick']),
                        }
                        episode_count += 1
                        for k, v in result.items():
                            results[k].append(v)
                        self._logger.info(
                            "[EVALUATOR] env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, result['reward'], episode_count
                            )
                        )
                if self._env_manager.done:
                    break

        duration = self._timer.value
        episode_reward = results['reward']
        envstep_count = np.sum(results['step'])
        success_count = np.sum(results['success'])
        success_rate = 0 if episode_count == 0 else success_count / episode_count
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': duration / n_episode,
            'success_rate': success_rate,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
        }
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        if self._tb_logger is not None:
            for k, v in info.items():
                if k in ['train_iter', 'ckpt_name', 'each_reward']:
                    continue
                if not np.isscalar(v):
                    continue
                self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
                self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)

        if success_rate > self._max_success_rate:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_success_rate = success_rate
        stop_flag = success_rate > self._stop_rate and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[EVALUATOR] " +
                "Current success rate: {} is greater than stop rate: {}".format(success_rate, self._stop_rate) +
                ", so the training is converged."
            )
        return stop_flag, success_rate
