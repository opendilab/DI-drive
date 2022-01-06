import logging

import numpy as np

from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math_utils import not_zero, wrap_to_pi, point_distance
from metadrive.utils.scene_utils import is_same_lane_index, is_following_lane_index
from metadrive.engine.core.manual_controller import KeyboardController, SteeringWheelController
from metadrive.utils import clip
from metadrive.examples import expert
from metadrive.policy.env_input_policy import EnvInputPolicy
from direct.controls.InputState import InputState
from metadrive.engine.engine_utils import get_global_config


#from metadrive.policy.discrete_policy import ActionType, DiscreteMetaAction
class FrontBackObjects:

    def __init__(self, front_ret, back_ret, front_dist, back_dist):
        self.front_objs = front_ret
        self.back_objs = back_ret
        self.front_dist = front_dist
        self.back_dist = back_dist

    def left_lane_exist(self):
        return True if self.front_dist[0] is not None else False

    def right_lane_exist(self):
        return True if self.front_dist[-1] is not None else False

    def has_front_object(self):
        return True if self.front_objs[1] is not None else False

    def has_back_object(self):
        return True if self.back_objs[1] is not None else False

    def has_left_front_object(self):
        return True if self.front_objs[0] is not None else False

    def has_left_back_object(self):
        return True if self.back_objs[0] is not None else False

    def has_right_front_object(self):
        return True if self.front_objs[-1] is not None else False

    def has_right_back_object(self):
        return True if self.back_objs[-1] is not None else False

    def front_object(self):
        return self.front_objs[1]

    def left_front_object(self):
        return self.front_objs[0]

    def right_front_object(self):
        return self.front_objs[-1]

    def back_object(self):
        return self.back_objs[1]

    def left_back_object(self):
        return self.back_objs[0]

    def right_back_object(self):
        return self.back_objs[-1]

    def left_front_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.front_dist[0]

    def right_front_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.front_dist[-1]

    def front_min_distance(self):
        return self.front_dist[1]

    def left_back_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.back_dist[0]

    def right_back_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.back_dist[-1]

    def back_min_distance(self):
        return self.back_dist[1]

    @classmethod
    def get_find_front_back_objs(cls, objs, lane, position, max_distance, ref_lanes=None):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        if ref_lanes is not None:
            assert lane in ref_lanes
        idx = lane.index[-1]
        left_lane = ref_lanes[idx - 1] if idx > 0 and ref_lanes is not None else None
        right_lane = ref_lanes[idx + 1] if ref_lanes and idx + 1 < len(ref_lanes) is not None else None
        lanes = [left_lane, lane, right_lane]

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None]
        back_ret = [None, None, None]

        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in enumerate(lanes)]

        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                if obj.lane is lane:
                    long = lane.local_coordinates(obj.position)[0] - current_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < min_back_long[i]:
                        min_back_long[i] = abs(long)
                        back_ret[i] = obj
                        find_back_in_current_lane[i] = True

                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(obj.lane):
                    long = obj.lane.local_coordinates(obj.position)[0] + left_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                elif not find_back_in_current_lane[i] and obj.lane.is_previous_lane_of(lane):
                    long = obj.lane.length - obj.lane.local_coordinates(obj.position)[0] + current_long[i]
                    if min_back_long[i] > long:
                        min_back_long[i] = long
                        back_ret[i] = obj

        return cls(front_ret, back_ret, min_front_long, min_back_long)


# class ManualControllableIDMPolicy(IDMPolicy):
#     def __init__(self, *args, **kwargs):
#         super(ManualControllableIDMPolicy, self).__init__(*args, **kwargs)
#         self.manual_control_policy = ManualControlPolicy()

#     def act(self, agent_id):
#         if self.control_object is self.engine.current_track_vehicle and self.engine.global_config["manual_control"]\
#                 and not self.engine.current_track_vehicle.expert_takeover:
#             return self.manual_control_policy.act(agent_id)
#         else:
#             return super(ManualControllableIDMPolicy, self).act(agent_id)


class ManualMacroDiscretePolicy(BasePolicy):
    NORMAL_SPEED = 65  # 65
    ACC_FACTOR = 1.0

    def __init__(self, control_object, random_seed):
        super(ManualMacroDiscretePolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.inputs = InputState()
        self.inputs.watchWithModifiers('accelerate', 'w')
        self.inputs.watchWithModifiers('deccelerate', 's')
        self.inputs.watchWithModifiers('laneLeft', 'a')
        self.inputs.watchWithModifiers('laneRight', 'd')
        # self.heading_pid = PIDController(1.7, 0.01, 3.5)
        # self.lateral_pid = PIDController(0.3, .002, 0.05)

        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.2, .002, 0.3)
        self.DELTA_SPEED = 5
        self.DELTA = 10
        self.target_lane = self.get_neighboring_lanes()[1]
        self.target_speed = self.NORMAL_SPEED
        self.stop_label = True

    def act(self, *args, **kwargs):
        lanes = self.get_neighboring_lanes()
        if (self.control_object.arrive_destination and hasattr(self.control_object, 'macro_succ')):
            self.control_object.macro_succ = True
        if (self.control_object.crash_vehicle and hasattr(self.control_object, 'crash_vehicle')):
            self.control_object.macro_crash = True

        #print('vel: {}'.format(self.control_object.velocity))
        if (len(args) >= 2):
            macro_action = args[1]
            # print('macro_control: {}'.format(macro_action))
        #print('arg length: {}'.format(len(args)))
        # agent_id = args[0]
        # macro_action = args[1]
        #print('macro_control: {}'.format(macro_action))
        if macro_action != "Holdon" and macro_action is not None:
            self.stop_label = False
        if macro_action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif macro_action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif macro_action == "LANE_LEFT":
            left_lane = lanes[0]
            if left_lane is not None:
                self.target_lane = left_lane
        elif macro_action == "LANE_RIGHT":
            right_lane = lanes[2]
            if right_lane is not None:
                self.target_lane = right_lane
        elif macro_action == "IDLE":
            self.target_speed = self.NORMAL_SPEED
            current_lane = lanes[1]
            self.target_lane = current_lane
        else:
            pass
            # current_lane = lanes[1]
            # self.target_lane = current_lane

        steering = self.steering_control(self.target_lane)
        throtle_brake = self.speed_control(self.target_speed)
        throtle_brake = throtle_brake if self.stop_label is False else -1
        #print('throtle_brake: {}'.format(throtle_brake))
        return [steering, throtle_brake]

        # steering = 0.0
        # throtle_brake = 0.0
        # centre_lane = lanes[1]
        # #print(lanes)
        # target_lane = centre_lane
        # if centre_lane is None:
        #     return [steering, throtle_brake]
        # if self.inputs.isSet('accelerate'):
        #     throtle_brake = 1.0
        # elif self.inputs.isSet('deccelerate'):
        #     throtle_brake = -1.0
        # if self.inputs.isSet('laneLeft'):
        #     left_lane = lanes[0]
        #     if left_lane is None:
        #         pass
        #         #steering = self.steering_control(centre_lane)
        #     else:
        #         target_lane = left_lane
        #         #steering = self.steering_control(left_lane)
        # elif self.inputs.isSet('laneRight'):
        #     right_lane = lanes[2]
        #     if right_lane is None:
        #         pass
        #         #steering = self.steering_control(centre_lane)
        #     else:
        #         target_lane = right_lane
        #         #steering = self.steering_control(right_lane)
        # else:
        #     pass
        # steering = self.steering_control(target_lane)
        # return [steering, throtle_brake]

    def get_neighboring_lanes(self):
        ref_lanes = self.control_object.navigation.current_ref_lanes
        lane = self.control_object.lane
        if ref_lanes is not None:
            assert lane in ref_lanes
        #if lane.after_end(self.control_object.position):
        if self.after_end_of_lane(lane, self.control_object.position):
            if self.control_object.navigation.next_ref_lanes is not None:
                ref_lanes = self.control_object.navigation.next_ref_lanes
            #ref_lanes = self.control_object.navigation.next_ref_lanes
            for ref_lane in ref_lanes:
                if (self.target_lane.is_previous_lane_of(ref_lane)):
                    self.target_lane = ref_lane
        else:
            pass
        idx = lane.index[-1]
        left_lane = ref_lanes[idx - 1] if idx > 0 and ref_lanes is not None else None
        right_lane = ref_lanes[idx + 1] if idx + 1 < len(ref_lanes) and ref_lanes is not None else None
        lanes = [left_lane, lane, right_lane]
        return lanes

    def after_end_of_lane(self, lane, position):
        longitudinal, _ = lane.local_coordinates(position)
        return longitudinal > lane.length - lane.VEHICLE_LENGTH

    def follow_road(self) -> None:
        return

    def move_to_next_road(self):
        current_lanes = self.control_object.navigation.current_ref_lanes
        # if(self.control_object.arrive_destination and hasattr(self.control_object, 'macro_succ')):
        #     self.control_object.macro_succ = True
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane):
                    self.routing_target_lane = lane
                return True
            return False
        elif self.control_object.lane in current_lanes and self.routing_target_lane is not self.control_object.lane:
            self.routing_target_lane = self.control_object.lane
            return True
        else:
            return True

    def lane_change_policy(self):
        current_lanes = self.control_object.navigation.current_ref_lanes
        available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0
        if lane_num_diff > 0:
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = [i for i in range(len(next_lanes))]
            else:
                index_range = [i for i in range(lane_num_diff, len(current_lanes))]
            self.available_routing_index_range = index_range
            if self.routing_target_lane.index[-1] not in index_range:
                if self.routing_target_lane.index[-1] > index_range[-1]:
                    return current_lanes[self.routing_target_lane.index[-1] - 1]
                else:
                    return current_lanes[self.routing_target_lane.index[-1] - 1]

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance control
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(wrap_to_pi(lane_heading - v_heading)) * 1.5
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def speed_control(self, target_speed):
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(target_speed, 0.001)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        return acceleration

    def acceleration(self, front_obj, dist_to_front) -> float:
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0)
        acceleration = self.ACC_FACTOR * (1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))
        if front_obj:
            d = dist_to_front
            speed_diff = self.desired_gap(ego_vehicle, front_obj) / not_zero(d)
            acceleration -= self.ACC_FACTOR * (speed_diff ** 2)
        return acceleration
