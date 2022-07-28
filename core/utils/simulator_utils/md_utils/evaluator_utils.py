from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner, NaiveReplayBuffer

from typing import Optional, Callable, Tuple
from collections import namedtuple
import numpy as np
import torch

from ding.envs import BaseEnvManager
from ding.torch_utils import to_tensor, to_ndarray
from ding.utils import build_logger, EasyTimer, SERIAL_EVALUATOR_REGISTRY
from ding.worker import ISerialEvaluator, VectorEvalMonitor


@SERIAL_EVALUATOR_REGISTRY.register('meta-interaction')
class MetadriveEvaluator(InteractionSerialEvaluator):
    """
    Overview:
        Interaction serial evaluator class, policy interacts with env.
    Interfaces:
        __init__, reset, reset_policy, reset_env, close, should_eval, eval
    Property:
        env, policy
    """

    def eval(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            n_episode: Optional[int] = None
    ) -> Tuple[bool, float]:
        '''
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_reward (:obj:`float`): Current eval_reward.
        '''
        z_success_times = 0
        z_fail_times = 0
        seq_traj_len = 10
        complete_ratio_list = []
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        envstep_count = 0
        info = {}
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        self._policy.reset()

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs)
                actions = {i: a['action'] for i, a in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, t in timesteps.items():
                    if t.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._policy.reset([env_id])
                        continue
                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['final_eval_reward']
                        arrive_dest = t.info['arrive_dest']
                        seq_traj_len = t.info['seq_traj_len']
                        seq_traj_len = int(seq_traj_len.numpy())
                        if arrive_dest:
                            z_success_times += 1
                        else:
                            z_fail_times += 1
                        if 'complete_ratio' in t.info:
                            complete_ratio_list.append(float(t.info['complete_ratio']))
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                            )
                        )
                    envstep_count += 1
        duration = self._timer.value
        episode_reward = eval_monitor.get_episode_reward()
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': n_episode,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': n_episode / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
            'reward_min': np.min(episode_reward),
            # 'each_reward': episode_reward,
            # 'zt env step': seq_traj_len * envstep_count,
        }
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            info.update(episode_info)
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        # self._logger.info(self._logger.get_tabulate_vars(info))
        for k, v in info.items():
            if k in ['train_iter', 'ckpt_name', 'each_reward']:
                continue
            if not np.isscalar(v):
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
        eval_reward = np.mean(episode_reward)
        if eval_reward > self._max_eval_reward:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_eval_reward = eval_reward
        stop_flag = eval_reward >= self._stop_value and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[DI-engine serial pipeline] " +
                "Current eval_reward: {} is greater than stop_value: {}".format(eval_reward, self._stop_value) +
                ", so your RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
            )
        print('zt')
        print('success times: {}'.format(z_success_times))
        print('fail_times: {}'.format(z_fail_times))
        success_ratio = float(z_success_times) / float(z_success_times + z_fail_times)
        print('success ratio: {}'.format(success_ratio))
        if len(complete_ratio_list) > 0:
            average_complete_ratio = np.mean(complete_ratio_list)
            print('average complete ratio: {}'.format(average_complete_ratio))
            print(complete_ratio_list)
        self._tb_logger.add_scalar('episode_info/succ_rate_iter', success_ratio, train_iter)
        self._tb_logger.add_scalar('episode_info/succ_rate_step', success_ratio, envstep * seq_traj_len)
        self._tb_logger.add_scalar('episode_info/complete_ratio_mean_iter', np.mean(complete_ratio_list), train_iter)
        self._tb_logger.add_scalar(
            'episode_info/complete_ratio_mean_step', np.mean(complete_ratio_list), envstep * seq_traj_len
        )
        self._tb_logger.add_scalar('episode_info/complete_ratio_std_iter', np.std(complete_ratio_list), train_iter)
        self._tb_logger.add_scalar(
            'episode_info/complete_ratio_std_step', np.std(complete_ratio_list), envstep * seq_traj_len
        )
        # print('zt env step: {}'.format(envstep * float(seq_traj_len)))
        # print('env step: {}'.format(envstep))
        # print('seq len : {}'.format(seq_traj_len))
        # self._tb_logger.add_scalar('succ_rate/', success_ratio, train_iter)
        # self._tb_logger.add_scalar('succ_rate/', success_ratio, envstep)

        return stop_flag, eval_reward

    def eval_verbose(
            self,
            save_ckpt_fn: Callable = None,
            train_iter: int = -1,
            envstep: int = -1,
            total_episode_count=-1,
            total_duration=-1,
            n_episode: Optional[int] = None
    ) -> Tuple[bool, float]:
        '''
        Overview:
            Evaluate policy and store the best policy based on whether it reaches the highest historical reward.
        Arguments:
            - save_ckpt_fn (:obj:`Callable`): Saving ckpt function, which will be triggered by getting the best reward.
            - train_iter (:obj:`int`): Current training iteration.
            - envstep (:obj:`int`): Current env interaction step.
            - n_episode (:obj:`int`): Number of evaluation episodes.
        Returns:
            - stop_flag (:obj:`bool`): Whether this training program can be ended.
            - eval_reward (:obj:`float`): Current eval_reward.
        '''
        z_success_times = 0
        z_fail_times = 0
        seq_traj_len = 10
        complete_ratio_list = []
        if n_episode is None:
            n_episode = self._default_n_episode
        assert n_episode is not None, "please indicate eval n_episode"
        envstep_count = 0
        info = {}
        eval_monitor = VectorEvalMonitor(self._env.env_num, n_episode)
        self._env.reset()
        self._policy.reset()

        with self._timer:
            while not eval_monitor.is_finished():
                obs = self._env.ready_obs
                obs = to_tensor(obs, dtype=torch.float32)
                policy_output = self._policy.forward(obs)
                actions = {i: a['action'] for i, a in policy_output.items()}
                actions = to_ndarray(actions)
                timesteps = self._env.step(actions)
                timesteps = to_tensor(timesteps, dtype=torch.float32)
                for env_id, t in timesteps.items():
                    if t.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._policy.reset([env_id])
                        continue
                    if t.done:
                        # Env reset is done by env_manager automatically.
                        self._policy.reset([env_id])
                        reward = t.info['final_eval_reward']
                        arrive_dest = t.info['arrive_dest']
                        seq_traj_len = t.info['seq_traj_len']
                        seq_traj_len = int(seq_traj_len.numpy())
                        if arrive_dest:
                            z_success_times += 1
                        else:
                            z_fail_times += 1
                        if 'complete_ratio' in t.info:
                            complete_ratio_list.append(float(t.info['complete_ratio']))
                        if 'episode_info' in t.info:
                            eval_monitor.update_info(env_id, t.info['episode_info'])
                        eval_monitor.update_reward(env_id, reward)
                        self._logger.info(
                            "[EVALUATOR]env {} finish episode, final reward: {}, current episode: {}".format(
                                env_id, eval_monitor.get_latest_reward(env_id), eval_monitor.get_current_episode()
                            )
                        )
                    envstep_count += 1
        duration = self._timer.value
        episode_reward = eval_monitor.get_episode_reward()
        info = {
            'train_iter': train_iter,
            'ckpt_name': 'iteration_{}.pth.tar'.format(train_iter),
            'episode_count': n_episode,
            'envstep_count': envstep_count,
            'avg_envstep_per_episode': envstep_count / n_episode,
            'evaluate_time': duration,
            'avg_envstep_per_sec': envstep_count / duration,
            'avg_time_per_episode': n_episode / duration,
            'reward_mean': np.mean(episode_reward),
            'reward_std': np.std(episode_reward),
            'reward_max': np.max(episode_reward),
            'reward_min': np.min(episode_reward),
            # 'each_reward': episode_reward,
            # 'zt env step': seq_traj_len * envstep_count,
        }
        episode_info = eval_monitor.get_episode_info()
        if episode_info is not None:
            info.update(episode_info)
        self._logger.info(self._logger.get_tabulate_vars_hor(info))
        # self._logger.info(self._logger.get_tabulate_vars(info))
        for k, v in info.items():
            if k in ['train_iter', 'ckpt_name', 'each_reward']:
                continue
            if not np.isscalar(v):
                continue
            self._tb_logger.add_scalar('{}_iter/'.format(self._instance_name) + k, v, train_iter)
            self._tb_logger.add_scalar('{}_step/'.format(self._instance_name) + k, v, envstep)
        eval_reward = np.mean(episode_reward)
        if eval_reward > self._max_eval_reward:
            if save_ckpt_fn:
                save_ckpt_fn('ckpt_best.pth.tar')
            self._max_eval_reward = eval_reward
        stop_flag = eval_reward >= self._stop_value and train_iter > 0
        if stop_flag:
            self._logger.info(
                "[DI-engine serial pipeline] " +
                "Current eval_reward: {} is greater than stop_value: {}".format(eval_reward, self._stop_value) +
                ", so your RL agent is converged, you can refer to 'log/evaluator/evaluator_logger.txt' for details."
            )
        print('zt')
        print('success times: {}'.format(z_success_times))
        print('fail_times: {}'.format(z_fail_times))
        success_ratio = float(z_success_times) / float(z_success_times + z_fail_times)
        print('success ratio: {}'.format(success_ratio))
        if len(complete_ratio_list) > 0:
            average_complete_ratio = np.mean(complete_ratio_list)
            print('average complete ratio: {}'.format(average_complete_ratio))
            print(complete_ratio_list)
        self._tb_logger.add_scalar('episode_info/succ_rate_iter', success_ratio, train_iter)
        self._tb_logger.add_scalar('episode_info/succ_rate_step', success_ratio, envstep * seq_traj_len)
        self._tb_logger.add_scalar('episode_info/succ_rate_episode', success_ratio, total_episode_count)
        self._tb_logger.add_scalar('episode_info/succ_rate_walltime', success_ratio, int(total_duration))
        self._tb_logger.add_scalar('episode_info/complete_ratio_mean_iter', np.mean(complete_ratio_list), train_iter)
        self._tb_logger.add_scalar(
            'episode_info/complete_ratio_mean_episode', np.mean(complete_ratio_list), total_episode_count
        )
        self._tb_logger.add_scalar(
            'episode_info/complete_ratio_mean_walltime', np.mean(complete_ratio_list), int(total_duration)
        )
        self._tb_logger.add_scalar(
            'episode_info/complete_ratio_mean_step', np.mean(complete_ratio_list), envstep * seq_traj_len
        )
        self._tb_logger.add_scalar('episode_info/complete_ratio_std_iter', np.std(complete_ratio_list), train_iter)
        self._tb_logger.add_scalar(
            'episode_info/complete_ratio_std_episode', np.std(complete_ratio_list), total_episode_count
        )
        self._tb_logger.add_scalar(
            'episode_info/complete_ratio_std_walltime', np.std(complete_ratio_list), int(total_duration)
        )
        self._tb_logger.add_scalar(
            'episode_info/complete_ratio_std_step', np.std(complete_ratio_list), envstep * seq_traj_len
        )
        # print('zt env step: {}'.format(envstep * float(seq_traj_len)))
        # print('env step: {}'.format(envstep))
        # print('seq len : {}'.format(seq_traj_len))
        # self._tb_logger.add_scalar('succ_rate/', success_ratio, train_iter)
        # self._tb_logger.add_scalar('succ_rate/', success_ratio, envstep)
        #print('zt episode: {}'.format(total_episode_count))

        return stop_flag, eval_reward
