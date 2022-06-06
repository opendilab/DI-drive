import copy
import numpy as np
from typing import Union, Dict, AnyStr, Tuple

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils import Config, safe_clip_for_small_array
from core.utils.simulator_utils.md_utils.navigation_utils import TrajNodeNavigation
from metadrive.component.vehicle.vehicle_type import DefaultVehicle


class MDDefaultVehicle(DefaultVehicle):

    def __init__(self, vehicle_config: Union[dict, Config] = None, name: str = None, random_seed=None):
        super(MDDefaultVehicle, self).__init__(vehicle_config, name, random_seed)
        self.macro_succ = False
        self.macro_crash = False
        self.last_spd = 0
        self.last_macro_position = self.last_position
        self.v_wps = [[0, 0], [1, 1]]
        self.v_indx = 1
        self.physics_world_step_size = self.engine.global_config["physics_world_step_size"]
        self.penultimate_state = {}

        self.penultimate_state['position'] = np.array([0, 0])  # self.last_position
        self.penultimate_state['yaw'] = 0
        self.penultimate_state['speed'] = 0
        self.traj_wp_list = []
        self.traj_wp_list.append(copy.deepcopy(self.penultimate_state))
        self.traj_wp_list.append(copy.deepcopy(self.penultimate_state))

        self.pen_state = {}
        self.prev_state = {}
        self.curr_state = {}
        self.pen_state['position'] = np.array([0, 0])  # self.last_position
        self.pen_state['yaw'] = 0
        self.pen_state['speed'] = 0
        self.prev_state['position'] = np.array([0, 0])  # self.last_position
        self.prev_state['yaw'] = 0
        self.prev_state['speed'] = 0
        self.curr_state['position'] = np.array([0, 0])  # self.last_position
        self.curr_state['yaw'] = 0
        self.curr_state['speed'] = 0

    def before_macro_step(self, macro_action):
        if macro_action is not None:
            self.last_macro_position = self.position
        else:
            pass
        return

    def reset_state_stack(self):
        self.pen_state['position'] = np.array([0, 0])  # self.last_position
        self.pen_state['yaw'] = 0
        self.pen_state['speed'] = 0
        self.prev_state['position'] = np.array([0, 0])  # self.last_position
        self.prev_state['yaw'] = 0
        self.prev_state['speed'] = 0
        self.curr_state['position'] = np.array([0, 0])  # self.last_position
        self.curr_state['yaw'] = 0
        self.curr_state['speed'] = 0

    def add_navigation(self):
        # if not self.config["need_navigation"]:
        #     return
        # navi = NodeNetworkNavigation if self.engine.current_map.road_network_type == NodeRoadNetwork \
        #     else EdgeNetworkNavigation
        navi = TrajNodeNavigation
        # print('seq len len ')
        # print(self.engine.global_config["seq_traj_len"])
        #print('navigation rigister')
        self.navigation = navi(
            self.engine,
            show_navi_mark=self.engine.global_config["vehicle_config"]["show_navi_mark"],
            random_navi_mark_color=self.engine.global_config["vehicle_config"]["random_navi_mark_color"],
            show_dest_mark=self.engine.global_config["vehicle_config"]["show_dest_mark"],
            show_line_to_dest=self.engine.global_config["vehicle_config"]["show_line_to_dest"],
            seq_traj_len=self.engine.global_config["seq_traj_len"],
            show_seq_traj=self.engine.global_config["show_seq_traj"]
        )


class MacroBaseVehicle(BaseVehicle):

    def __init__(self, vehicle_config: Union[dict, Config] = None, name: str = None, random_seed=None):
        super(MacroBaseVehicle, self).__init__(vehicle_config, name, random_seed)
        self.macro_succ = False
        self.macro_crash = False
