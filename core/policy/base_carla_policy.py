from collections import namedtuple, deque
from typing import List, Dict, Optional, Union, Any, NamedTuple

from core.utils.others.config_helper import deep_merge_dicts
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

    total_field = set(['learn', 'collect', 'eval'])

    def __init__(self, cfg: dict, model: Any = None, enable_field: Optional[List[str]] = None) -> None:
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        if enable_field is None:
            self._enable_field = self.total_field
        else:
            self._enable_field = enable_field
        self._model = model

        for field in self._enable_field:
            getattr(self, '_init_' + field)()

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

    def _create_model(self, cfg: dict, model: Any) -> Any:
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
