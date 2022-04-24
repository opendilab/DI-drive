from metadrive.manager.agent_manager import AgentManager
from core.utils.simulator_utils.md_utils.macro_policy import ManualMacroDiscretePolicy
from metadrive.utils.space import ParameterSpace, VehicleParameterSpace
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.utils import Config, safe_clip_for_small_array
from typing import Union, Dict, AnyStr, Tuple


class MacroDefaultVehicle(DefaultVehicle):

    def __init__(self, vehicle_config: Union[dict, Config] = None, name: str = None, random_seed=None):
        super(MacroDefaultVehicle, self).__init__(vehicle_config, name, random_seed)
        self.macro_succ = False
        self.macro_crash = False
        self.last_macro_position = self.last_position

    def before_macro_step(self, macro_action):
        if macro_action is not None:
            self.last_macro_position = self.position
        else:
            pass
        return


class MacroAgentManager(AgentManager):

    def _get_policy(self, obj):
        policy = ManualMacroDiscretePolicy(obj, self.generate_seed())
        return policy

    def before_step(self, external_actions=None):
        # not in replay mode
        self._agents_finished_this_frame = dict()
        step_infos = {}
        for agent_id in self.active_agents.keys():
            policy = self.engine.get_policy(self._agent_to_object[agent_id])
            macro_action = None
            if external_actions is not None and agent_id in external_actions.keys():
                macro_action = external_actions[agent_id]
            action = policy.act(agent_id, macro_action)
            #action = policy.act(agent_id, external_actions)
            step_infos[agent_id] = policy.get_action_info()
            step_infos[agent_id].update(self.get_agent(agent_id).before_step(action))
            self.get_agent(agent_id).before_macro_step(macro_action)

        finished = set()
        for v_name in self._dying_objects.keys():
            self._dying_objects[v_name][1] -= 1
            if self._dying_objects[v_name][1] == 0:  # Countdown goes to 0, it's time to remove the vehicles!
                v = self._dying_objects[v_name][0]
                self._remove_vehicle(v)
                finished.add(v_name)
        for v_name in finished:
            self._dying_objects.pop(v_name)
        return step_infos

    def _get_vehicles(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        ret = {}
        # v_type = random_vehicle_type(self.np_random) if self.engine.global_config["random_agent_model"] else \
        #     vehicle_type[self.engine.global_config["vehicle_config"]["vehicle_model"]]
        v_type = MacroDefaultVehicle
        for agent_id, v_config in config_dict.items():
            obj = self.spawn_object(v_type, vehicle_config=v_config)
            ret[agent_id] = obj
            policy = self._get_policy(obj)
            self.engine.add_policy(obj.id, policy)
        return ret
