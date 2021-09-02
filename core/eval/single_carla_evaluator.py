import os
import torch
import numpy as np
from typing import Any, Dict, List

from .base_evaluator import BaseEvaluator
from ding.torch_utils.data_helper import to_tensor


class SingleCarlaEvaluator(BaseEvaluator):
    """
    Carla envaluator used to evaluate a single environment. It is mainly used to visualize the
    evaluation results. It uses a environment in DI-engine form and can be rendered in the runtime.

    :Arguments:
            - cfg (Dict): Config dict
            - env (Any): Carla env, should be in DI-engine form
            - policy (Any): the policy to pe evaluated

    :Interfaces: reset, eval, close, should_eval
    """

    config = dict(
        render=False,
        eval_freq=1000,
        transform_obs=False,
    )

    def __init__(self, cfg: Dict, env: Any, policy: Any) -> None:
        super().__init__(cfg, env, policy)
        self._render = self._cfg.render
        self._transform_obs = self._cfg.transform_obs
        self._last_eval_iter = 0

    def close(self) -> None:
        """
        Close evaluator. It will close the EnvManager
        """
        self._env.close()

    def reset(self) -> None:
        pass

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

    def eval(self, reset_param: Dict = None) -> float:
        """
        Running one episode evaluation with provided reset params.

        :Arguments:
            - reset_param (Dict, optional): Reset parameter for environment. Defaults to None.

        :Returns:
            float: Evaluation final reward.
        """
        self._policy.reset([0])
        eval_reward = 0
        success = False
        if reset_param is not None:
            obs = self._env.reset(**reset_param)
        else:
            obs = self._env.reset()

        with self._timer:
            while True:
                if self._transform_obs:
                    obs = to_tensor(obs, dtype=torch.float32)
                actions = self._policy.forward({0: obs})
                action = actions[0]['action']
                timestep = self._env.step(action)
                obs = timestep.obs
                if self._render:
                    self._env.render()
                if timestep.info.get('abnormal', False):
                    # If there is an abnormal timestep, reset all the related variables(including this env).
                    self._policy.reset(**reset_param)
                    action = np.array([0.0, 0.0, 0.0])
                    timestep = self._env.step(action)

                if timestep.done:
                    eval_reward = timestep.info['final_eval_reward']
                    success = timestep.info['success']
                    break

        duration = self._timer.value
        info = {
            'evaluate_time': duration,
            'eval_reward': eval_reward,
            'success': success,
        }
        print(
            "[EVALUATOR] Evaluation ends:\n{}".format(
                '\n'.join(['\t{}: {:.3f}'.format(k, v) for k, v in info.items()])
            )
        )
        print("[EVALUATOR] Evaluate done!")
        return success
