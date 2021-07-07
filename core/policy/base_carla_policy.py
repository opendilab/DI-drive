from collections import namedtuple, deque
from typing import List, Dict, Optional, Union, Any, NamedTuple

from ding.policy import Policy


class BaseCarlaPolicy(Policy):
    """
    Base class for Carla policy interact with environments. The policy is defined in standard DI-engine form which
    has several modes to change its running form, and can interact with several environments controlled by a
    ``EnvManager``. The policy is designed to support Supervised Learning, Reinforcement Learning and other method
    as well as expert policy, each may have different kinds of interfaces and modes.

    By default, it has 3 modes: `learn`, `collect` and `eval`. To set policy to a specific mode, call the policy
    with ``policy.xxx_mode``. Then all the supported interfaces can be defined in ``_interface_xxx`` or ``_interfaces``
    method. For example, calling ``policy.collect_mode.forward`` is equal to calling ``policy._forward_collect``.
    Some mode-specific interfaces may be defined specially by user.

    :Interfaces: init, forward, reset, process_transition, get_train_sample
    """

    config = dict()

    def __init__(self, cfg: Dict) -> None:
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg.update(cfg)
        else:
            self._cfg = cfg

    def _init_learn(self) -> None:
        pass

    def _forward_learn(self, data: Dict) -> Dict[str, Any]:
        pass

    def _init_collect(self) -> None:
        pass

    def _forward_collect(self, data_id: List[int], data: Dict, **kwargs) -> Dict:
        pass

    def _init_eval(self) -> None:
        pass

    def _forward_eval(self, data_id: List[int], data: Dict) -> Dict[str, Any]:
        pass

    def _process_transition(self, obs: Any, model_output: Dict, timestep: NamedTuple) -> Dict[str, Any]:
        transition = {
            'obs': obs,
            'action': model_output,
        }
        return transition

    def _get_train_sample(self, data: Any) -> Optional[List]:
        if isinstance(data, deque):
            data = list(data)
        return data
