import os
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
from itertools import product
import torch
from typing import Dict, Any, List, NamedTuple, Optional

from .base_evaluator import BaseEvaluator
from core.data.benchmark import ALL_SUITES
from core.data.benchmark.benchmark_utils import get_suites_list, gather_results, read_pose_txt, get_benchmark_dir
from ding.envs import BaseEnvManager
from ding.torch_utils.data_helper import to_tensor


class CarlaBenchmarkEvaluator(BaseEvaluator):
    """
    Evaluator used to evaluate a policy with Carla benchmark evaluation suites. It uses several environments
    in ``EnvManager`` to evaluate policy. For every suites provided by user, evaluator will first find and
    store all available reset params from Benchmark files and store them in a queue such that each reset
    param is evaluated once and only once. The evaluation results are stored in a '.csv' file with reward,
    success and failure status and reset param of the episode.

    Note:
        Env manager must run WITHOUT auto reset.

    :Arguments:
        - cfg (Dict): Config dict.
        - env (BaseEnvManager): Env manager used to evaluate.
        - policy (Any): Policy to evaluate. Must have ``forward`` method.

    :Interfaces: reset, eval, close

    :Properties:
        - env (BaseEnvManager): Env manager with several environments used to evaluate.
        - policy (Any): Policy instance to interact with envs.
    """

    config = dict(
        benchmark_dir=None,
        result_dir='',
        transform_obs=False,
        episodes_per_suite=100,
        resume=False,
        suite='FullTown01-v0',
        weathers=None,
        seed=0,
    )

    def __init__(self, cfg: Dict, env: BaseEnvManager, policy: Any) -> None:
        super().__init__(cfg, env, policy)
        self._benchmark_dir = self._cfg.benchmark_dir
        self._result_dir = self._cfg.result_dir
        self._transform_obs = self._cfg.transform_obs
        self._episodes_per_suite = self._cfg.episodes_per_suite
        self._resume = self._cfg.resume
        if self._benchmark_dir is None:
            self._benchmark_dir = get_benchmark_dir()

        suite = self._cfg.suite
        self._eval_suite_list = get_suites_list(suite)
        self._seed = self._cfg.seed
        self._weathers = self._cfg.weathers
        self._close_flag = False

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
        Close the collector and the env manager if not closed.
        """
        if self._close_flag:
            return
        self._close_flag = True
        self._env_manager.close()

    def reset(self) -> None:
        """
        Reset evaluator and policies.
        """
        self._policy.reset([i for i in range(self._env_num)])

    def eval(
            self,
            policy_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Run evaluation with provided policy arguments. It will evaluate all available episodes of the benchmark suite
        unless `episode_per_suite` is set in config.

        :Arguments:
            - policy_kwargs (Dict, optional): Additional arguments in policy forward. Defaults to None.
        """
        total_time = 0.0
        if policy_kwargs is None:
            policy_kwargs = dict()
        if self._result_dir != '':
            os.makedirs(self._result_dir, exist_ok=True)
        self.reset()

        for suite in tqdm(self._eval_suite_list):
            args, kwargs = ALL_SUITES[suite]
            assert len(args) == 0
            reset_params = kwargs.copy()
            poses_txt = reset_params.pop('poses_txt')
            weathers = reset_params.pop('weathers')
            suite_name = suite + '_seed%d' % self._seed
            summary_csv = os.path.join(self._result_dir, suite_name + ".csv")
            if os.path.exists(summary_csv):
                summary = pd.read_csv(summary_csv)
            else:
                summary = pd.DataFrame()

            if self._weathers is not None:
                weathers = self._weathers

            pose_pairs = read_pose_txt(self._benchmark_dir, poses_txt)

            episode_queue = deque()
            running_env_params = dict()
            results = []
            running_envs = 0

            for episode, (weather, (start, end)) in enumerate(product(weathers, pose_pairs)):
                if episode >= self._episodes_per_suite:
                    break
                param = reset_params.copy()
                param['start'] = start
                param['end'] = end
                param['weather'] = weather
                if self._resume and len(summary) > 0 and ((summary['start'] == start) & (summary['end'] == end) &
                                                          (summary['weather'] == weather)).any():
                    print('[EVALUATOR]', weather, start, end, 'already exist')
                    continue
                if running_envs < self._env_num:
                    running_env_params[running_envs] = param
                    running_envs += 1
                else:
                    episode_queue.append(param)

            for env_id in running_env_params:
                self._env_manager.seed({env_id: self._seed})
            self._env_manager.reset(running_env_params)
            with self._timer:
                while True:
                    obs = self._env_manager.ready_obs
                    if self._transform_obs:
                        obs = to_tensor(obs, dtype=torch.float32)
                    policy_output = self._policy.forward(obs, **policy_kwargs)
                    actions = {env_id: output['action'] for env_id, output in policy_output.items()}
                    timesteps = self._env_manager.step(actions)
                    for i, t in timesteps.items():
                        if t.info.get('abnormal', False):
                            self._policy.reset([i])
                            self._env_manager.reset(reset_params={i: running_env_params[i]})
                            continue
                        if t.done:
                            self._policy.reset([i])
                            result = {
                                'start': running_env_params[i]['start'],
                                'end': running_env_params[i]['end'],
                                'weather': running_env_params[i]['weather'],
                                'success': t.info['success'],
                                'collided': t.info['collided'],
                                'timecost': int(t.info['tick']),
                            }
                            results.append(result)
                            if episode_queue:
                                reset_param = episode_queue.pop()
                                self._env_manager.reset({i: reset_param})
                                running_env_params[i] = reset_param
                    if self._env_manager.done:
                        break
            duration = self._timer.value
            total_time += duration
            summary = pd.DataFrame(results)
            summary.to_csv(summary_csv, index=False)

        results = gather_results(self._result_dir)
        print(results)
        print('[EVALUATOR] Total time: %.3f hours.' % (total_time / 3600.0))
