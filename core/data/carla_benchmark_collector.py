import os
import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional, Union
from itertools import product
import random

from .base_collector import BaseCollector
from core.data.benchmark import ALL_SUITES
from core.data.benchmark.benchmark_utils import get_suites_list, read_pose_txt, get_benchmark_dir
from ding.envs import BaseEnvManager
from ding.torch_utils.data_helper import to_ndarray


class CarlaBenchmarkCollector(BaseCollector):
    """
    Collector to collect Carla benchmark data with envs. It uses several environments in ``EnvManager`` to
    collect data. It will automatically get params to reset environments. For every suite provided by user,
    collector will find all available reset params from benchmark files and store them in a list. When
    collecting data, the collector will collect each suite in average and store the index of each suite,
    to make sure each reset param is collected once and only once. The collected data are stored in a
    trajectory list, with observations, actions and reset param of the episode.

    Note:
        Env manager must run WITHOUT auto reset.

    :Arguments:
        - cfg (Dict): Config dict.
        - env (BaseEnvManager): Env manager used to collect data.
        - policy (Any): Policy used to collect data. Must have ``forward`` method.

    :Interfaces: reset, collect, close

    :Properties:
        - env (BaseEnvManager): Env manager with several environments used to sample data.
        - policy (Any): Policy instance to interact with envs.
    """

    config = dict(
        benchmark_dir=None,
        # suite name, can be str or list
        suite='FullTown01-v0',
        seed=None,
        # whether make seed of each env different
        dynamic_seed=True,
        # manually set weathers rather than read from suite
        weathers=None,
        # whether apply hard failure judgement in suite
        # by default in benchmark, collided will not cause failure
        nocrash=False,
        # whether shuffle env setting in suite
        shuffle=False,
    )

    def __init__(
            self,
            cfg: Dict,
            env: BaseEnvManager,
            policy: Any,
    ) -> None:
        super().__init__(cfg, env, policy)
        self._benchmark_dir = self._cfg.benchmark_dir
        suite = self._cfg.suite
        self._seed = self._cfg.seed
        self._dynamic_seed = self._cfg.dynamic_seed
        self._weathers = self._cfg.weathers
        self._shuffle = self._cfg.shuffle
        if self._benchmark_dir is None:
            self._benchmark_dir = get_benchmark_dir()
        self._collect_suite_list = get_suites_list(suite)
        print('[COLLECTOR] Find suites:', self._collect_suite_list)
        self._suite_num = len(self._collect_suite_list)
        self._close_flag = False
        self._collect_suite_reset_params = dict()
        self._collect_suite_index_dict = dict()

        self._traj_cache = {env_id: deque() for env_id in range(self._env_num)}
        self._obs_cache = [None for _ in range(self._env_num)]
        self._actions_cache = [None for _ in range(self._env_num)]

        self._generate_suite_reset_params()

    @property
    def env(self) -> BaseEnvManager:
        return self._env_manager

    @env.setter
    def env(self, _env_manager: BaseEnvManager) -> None:
        assert not _env_manager._auto_reset, "auto reset for env manager should be closed!"
        self._end_flag = False
        self._env_manager = _env_manager
        self._env_manager.launch()
        self._env_num = self._env_manager.env_num

    def close(self) -> None:
        """
        Close collector and env manager if not closed.
        """
        if self._end_flag:
            return
        self._collect_suite_reset_params.clear()
        self._collect_suite_index_dict.clear()
        self._env_manager.close()
        self._end_flag = True

    def _generate_suite_reset_params(self):
        for suite in self._collect_suite_list:
            self._collect_suite_reset_params[suite] = list()
            self._collect_suite_index_dict[suite] = 0
            args, kwargs = ALL_SUITES[suite]
            assert len(args) == 0
            reset_params = kwargs.copy()
            poses_txt = reset_params.pop('poses_txt')
            weathers = reset_params.pop('weathers')
            if self._weathers is not None:
                weathers = self._weathers
            pose_pairs = read_pose_txt(self._benchmark_dir, poses_txt)
            for weather, (start, end) in product(weathers, pose_pairs):
                param = reset_params.copy()
                param['start'] = start
                param['end'] = end
                param['weather'] = weather
                if self._cfg.nocrash:
                    param['col_is_failure'] = True
                self._collect_suite_reset_params[suite].append(param)
            if self._shuffle:
                random.shuffle(self._collect_suite_reset_params[suite])

    def reset(self, suite: Union[List, str] = None) -> None:
        """
        Reset collector and policies. Clear data cache storing data trajectories. If 'suite' is provided
        in arguments, the collector will change its collected suites and generate reset params again.

        :Arguments:
            - suite (Union[List, str], optional): Collected suites after reset. Defaults to None.
        """
        for env_id in range(self._env_num):
            self._traj_cache[env_id].clear()
        self._policy.reset([i for i in range(self._env_num)])
        self._end_flag = False

        if suite is not None:
            self._collect_suite_reset_params.clear()
            self._collect_suite_index_dict.clear()
            self._collect_suite_list = get_suites_list(suite)
            print('[COLLECTOR] Find suites:', [s for s in self._collect_suite_list])
            self._suite_num = len(self._collect_suite_list)
            self._generate_suite_reset_params()

    def collect(
            self,
            n_episode: int,
            policy_kwargs: Optional[Dict] = None,
    ) -> List:
        """
        Collect data from policy and env manager. It will collect each benchmark suite in average
        according to 'n_episode'.

        :Arguments:
            - n_episode (int): Num of episodes to collect.
            - policy_kwargs (Dict, optional): Additional arguments in policy forward. Defaults to None.

        :Returns:
            List: List of collected data. Each elem stores an episode trajectory.
        """
        if policy_kwargs is None:
            policy_kwargs = dict()
        assert len(self._collect_suite_list) > 0, self._collect_suite_list

        if n_episode < self._env_num:
            print("[WARNING] Number of envs larger than number of episodes. May waste resource")

        for env_id in range(self._env_num):
            self._traj_cache[env_id].clear()
        self._policy.reset([i for i in range(self._env_num)])

        running_env_params = dict()
        running_envs = 0
        prepare_enough = False

        while not prepare_enough:
            for suite in self._collect_suite_list:
                suite_index = self._collect_suite_index_dict[suite]
                suite_params = self._collect_suite_reset_params[suite]
                reset_param = suite_params[suite_index]
                if running_envs < self._env_num and running_envs < n_episode:
                    running_env_params[running_envs] = reset_param
                    running_envs += 1
                    self._collect_suite_index_dict[suite] += 1
                    self._collect_suite_index_dict[suite] %= len(suite_params)
                else:
                    prepare_enough = True
                    break

        if self._seed is not None:
            # dynamic seed: different seed for each env
            if self._dynamic_seed:
                self._env_manager.seed(self._seed)
            else:
                for env_id in running_env_params:
                    self._env_manager.seed({env_id: self._seed})
        self._env_manager.reset(running_env_params)

        return_data = []
        env_fail_times = {env_id: 0 for env_id in running_env_params}
        collected_episodes = running_envs - 1
        collected_samples = 0

        with self._timer:
            while True:
                obs = self._env_manager.ready_obs
                env_ids = list(obs.keys())
                for env_id in env_ids:
                    if env_id not in running_env_params:
                        obs.pop(env_id)
                if len(obs) == 0:
                    break
                policy_output = self._policy.forward(obs, **policy_kwargs)
                actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                actions = to_ndarray(actions)
                for env_id in actions:
                    self._obs_cache[env_id] = obs[env_id]
                    self._actions_cache[env_id] = actions[env_id]
                timesteps = self._env_manager.step(actions)
                for env_id, timestep in timesteps.items():
                    if timestep.info.get('abnormal', False):
                        # If there is an abnormal timestep, reset all the related variables(including this env).
                        self._traj_cache[env_id].clear()
                        self._policy.reset([timestep])
                        self._env_manager.reset(reset_param={env_id: running_env_params[env_id]})
                        print('[COLLECTOR] env_id abnormal step', env_id, timestep.info)
                        continue
                    transition = self._policy.process_transition(
                        self._obs_cache[env_id], self._actions_cache[env_id], timestep
                    )
                    self._traj_cache[env_id].append(transition)
                    if timestep.done:
                        if timestep.info['success'] and len(self._traj_cache[env_id]) > 50:
                            env_fail_times[env_id] = 0
                            env_param = running_env_params[env_id]
                            episode_data = {'env_param': env_param, 'data': list(self._traj_cache[env_id])}
                            return_data.append(episode_data)
                            collected_samples += len(self._traj_cache[env_id])
                            collected_episodes += 1
                            if collected_episodes < n_episode:
                                suite_index = collected_episodes % self._suite_num
                                next_suite = self._collect_suite_list[suite_index]
                                reset_param_index = self._collect_suite_index_dict[next_suite]
                                reset_param = self._collect_suite_reset_params[next_suite][reset_param_index]
                                self._collect_suite_index_dict[next_suite] += 1
                                self._collect_suite_index_dict[next_suite] %= len(
                                    self._collect_suite_reset_params[next_suite]
                                )
                                running_env_params[env_id] = reset_param
                                self._env_manager.reset({env_id: reset_param})
                        else:
                            env_fail_times[env_id] += 1
                            info = timestep.info
                            for k in list(info.keys()):
                                if 'reward' in k:
                                    info.pop(k)
                                if k in ['timestamp']:
                                    info.pop(k)
                            print('[COLLECTOR] env_id {} not success'.format(env_id), info)
                            if env_fail_times[env_id] < 5:
                                # not reach max fail times, continue reset param
                                reset_param = running_env_params[env_id]
                            else:
                                # reach max fail times, skip to next reset param
                                env_fail_times[env_id] = 0
                                suite_index = collected_episodes % self._suite_num
                                next_suite = self._collect_suite_list[suite_index]
                                reset_param_index = self._collect_suite_index_dict[next_suite]
                                reset_param = self._collect_suite_reset_params[next_suite][reset_param_index]
                                self._collect_suite_index_dict[next_suite] += 1
                                self._collect_suite_index_dict[next_suite] %= len(
                                    self._collect_suite_reset_params[next_suite]
                                )
                            running_env_params[env_id] = reset_param
                            self._env_manager.reset({env_id: reset_param})
                        self._traj_cache[env_id].clear()
                        self._policy.reset([env_id])
                if self._env_manager.done:
                    break

        duration = self._timer.value
        print("[COLLECTOR] Finish collection, time cost: {:.2f}s, total frames: {}".format(duration, collected_samples))

        return return_data
