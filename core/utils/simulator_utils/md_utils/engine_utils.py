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
