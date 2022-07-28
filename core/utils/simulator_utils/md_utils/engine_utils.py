from metadrive.engine.base_engine import BaseEngine
import copy
import logging

from typing import Callable, Optional, Union, List, Dict, AnyStr


class MacroEngine(BaseEngine):

    def before_step_macro(self, actions=None) -> Dict:
        """
        Update states after finishing movement
        :return: if this episode is done
        """
        step_infos = {}
        for manager in self._managers.values():
            if (manager.__class__.__name__ == 'MacroAgentManager'):
                step_infos.update(manager.before_step(actions))
            else:
                step_infos.update(manager.before_step())
        return step_infos


class TrajEngine(BaseEngine):

    def before_step_traj(self, frame=0, wps=None) -> Dict:
        """
        Update states after finishing movement
        :return: if this episode is done
        """
        step_infos = {}
        for manager in self._managers.values():
            if (manager.__class__.__name__ == 'TrajAgentManager'):
                step_infos.update(manager.before_step(frame, wps))
            else:
                step_infos.update(manager.before_step())
        return step_infos


def initialize_engine(env_global_config):
    cls = TrajEngine
    if cls.singleton is None:
        # assert cls.global_config is not None, "Set global config before initialization BaseEngine"
        cls.singleton = cls(env_global_config)
    else:
        raise PermissionError("There should be only one BaseEngine instance in one process")
    return cls.singleton


def get_engine():
    return TrajEngine.singleton


def get_object(object_name):
    return get_engine().get_objects([object_name])


def engine_initialized():
    return False if TrajEngine.singleton is None else True


def close_engine():
    if TrajEngine.singleton is not None:
        TrajEngine.singleton.close()
        TrajEngine.singleton = None


def get_global_config():
    engine = get_engine()
    return engine.global_config.copy()


def set_global_random_seed(random_seed: Optional[int]):
    """
    Update the random seed and random engine
    All subclasses of Randomizable will hold the same random engine, after calling this function
    :param random_seed: int, random seed
    """
    engine = get_engine()
    if engine is not None:
        engine.seed(random_seed)
    else:
        logging.warning("BaseEngine is not launched, fail to sync seed to engine!")
