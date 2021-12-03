import numpy as np
from typing import Dict, List, Tuple
import carla

from .basic_planner import AgentState, BasicPlanner, RoadOption
from core.utils.simulator_utils.carla_agents.navigation.types_behavior import Cautious, Aggressive, Normal
from core.simulators.carla_data_provider import CarlaDataProvider
from core.utils.simulator_utils.carla_agents.tools.misc import is_within_distance, compute_distance, positive
from core.utils.planner.planner_utils import get_next_until_junction, generate_change_lane_route


class BehaviorPlanner(BasicPlanner):
    """
    Behavior local planner for Carla simulator. It can set route the same way as ``BasicPlanner``. BehaviorPlanner can
    check the speed limitations, traffic lights in evironment, and also take nearby vehicles into account.
    Lane changing
    decisions can be taken by analyzing the surrounding environment, such as overtaking or tailgating avoidance.
    Besides, it can also keep safety distance from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Different sets of behaviors are encoded in the agent, from cautious to a more
    aggressive ones.

    :Arguments:
        - cfg (Dict): Config dict.

    :Interfaces: set_destination, set_route, run_step, get_waypoints_list, clean_up
    """

    config = dict(
        min_distance=5.0,
        resolution=5.0,
        fps=10,
        debug=False,
        behavior='normal',
        min_speed=5,
    )

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)

        self._speed = 0
        self._incoming_waypoint = None
        self._incoming_direction = None
        self._light_state = "Green"
        self._light_id_to_ignore = -1

        behavior = self._cfg.behavior
        self._min_speed = self._cfg.min_speed

        # Parameters for agent behavior
        if behavior == 'cautious':
            self.behavior = Cautious()
        elif behavior == 'normal':
            self.behavior = Normal()
        elif behavior == 'aggressive':
            self.behavior = Aggressive()
        else:
            raise ValueError(
                'behavior must in ["normal", "aggresive", and "cautious"], got {} instead.'.format(behavior)
            )

    def _behavior_is_vehicle_hazard(
            self,
            vehicle_list: List,
            proximity_th: float,
            up_angle_th: float,
            low_angle_th: float = 0,
            lane_offset: int = 0
    ) -> Tuple[bool, carla.Actor, float]:

        # Get the right offset
        if self.current_waypoint.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        for target_vehicle in vehicle_list:

            target_vehicle_loc = target_vehicle.get_location()
            # If the object is not in our next or current lane it's not an obstacle

            target_wpt = self._map.get_waypoint(target_vehicle_loc)
            if target_wpt.road_id != self.current_waypoint.road_id or \
                    target_wpt.lane_id != self.current_waypoint.lane_id + lane_offset:
                find_in_next = False
                for i in range(1, 5):
                    next_wpt = self.get_incoming_waypoint_and_direction(steps=i)[0]
                    if next_wpt is None:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or \
                            target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue
                    find_in_next = True

                if not find_in_next:
                    continue

            if is_within_distance(target_vehicle_loc, self._vehicle_location,
                                  self._hero_vehicle.get_transform().rotation.yaw, proximity_th, up_angle_th,
                                  low_angle_th):

                return (True, target_vehicle, compute_distance(target_vehicle_loc, self._vehicle_location))

        return (False, None, -1)

    def change_route_in_current_road(self, route):
        route_end = route[-1][0]
        find_in_buffer = False
        wpt_index = -1
        if self._waypoints_buffer:
            for i, (waypoint, _) in enumerate(self._waypoints_buffer):
                if waypoint.road_id != route_end.road_id:
                    wpt_index = i
                    find_in_buffer = True
                    break
        if find_in_buffer:
            for i in range(wpt_index):
                self._waypoints_buffer.popleft()
                self.distance_to_goal -= self.distances.popleft()
        else:
            for i in range(len(self._waypoints_buffer)):
                self._waypoints_buffer.popleft()
                if self.distances:
                    self.distance_to_goal -= self.distances.popleft()
            for i, (waypoint, _) in enumerate(self._waypoints_queue):
                if waypoint.road_id != route_end.road_id:
                    wpt_index = i
                    break
            if wpt_index >= 0:
                for i in range(wpt_index):
                    self._waypoints_queue.popleft()
                    if self.distances:
                        self.distance_to_goal -= self.distances.popleft()
            else:
                self._waypoints_queue.clear()
                self.distances.clear()
                self.distance_to_goal = 0
                self._waypoints_queue.append((self.end_waypoint, RoadOption.LANEFOLLOW))
        if self._waypoints_buffer:
            old_route_start = self._waypoints_buffer[0][0]
        else:
            old_route_start = self._waypoints_queue[0][0]
        link_route = self._grp.trace_route(route_end.transform.location, old_route_start.transform.location)
        self.add_route_in_front(link_route)
        self.add_route_in_front(route)

    def run_step(self) -> None:
        """
        Run one step of local planner. It will update node and target waypoint and road option, and check agent
        states. Besides, it may have some behavior like overtaking, tailgating, safe distance keeping and so on
        """
        assert self._route is not None

        vehicle_transform = CarlaDataProvider.get_transform(self._hero_vehicle)
        self._vehicle_location = vehicle_transform.location
        self.current_waypoint = self._map.get_waypoint(
            self._vehicle_location, lane_type=carla.LaneType.Driving, project_to_road=True
        )

        # Add waypoints into buffer if empty
        if not self._waypoints_buffer:
            for i in range(min(self._buffer_size, len(self._waypoints_queue))):
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                else:
                    break

            # If no waypoints return with current waypoint
            if not self._waypoints_buffer:
                self.target_waypoint = self.current_waypoint
                self.node_waypoint = self.current_waypoint
                self.target_road_option = RoadOption.VOID
                self.node_road_option = RoadOption.VOID
                self.agent_state = AgentState.VOID
                return

        # Find the most far waypoint within min distance
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoints_buffer):
            cur_dis = waypoint.transform.location.distance(vehicle_transform.location)
            if cur_dis < self._min_distance:
                max_index = i
        if max_index >= 0:
            for i in range(max_index + 1):
                self.node_waypoint, self.node_road_option = self._waypoints_buffer.popleft()
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                if self.distances:
                    self.distance_to_goal -= self.distances.popleft()

        # Update information
        if self._waypoints_buffer:
            self.target_waypoint, self.target_road_option = self._waypoints_buffer[0]
        self.agent_state = AgentState.NAVIGATING
        self._speed = CarlaDataProvider.get_speed(self._hero_vehicle) * 3.6
        self.speed_limit = self._hero_vehicle.get_speed_limit()
        look_ahead_steps = int((self.speed_limit) / 10)
        incoming_waypoint, incoming_direction = self.get_incoming_waypoint_and_direction(look_ahead_steps)

        # 1: Red lights and stops behavior
        light_state, _ = CarlaDataProvider.is_light_red(self._hero_vehicle)
        if light_state:
            self.agent_state = AgentState.BLOCKED_RED_LIGHT
            return

        # 2.1: Pedestrian avoidancd behaviors
        walker_state, walker, w_distance = self._pedestrian_avoid_manager()

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                self._hero_vehicle.bounding_box.extent.y, self._hero_vehicle.bounding_box.extent.x
            )

            # Emergency brake if the car is very close.
            if distance < self.behavior.braking_distance:
                self.agent_state = AgentState.BLOCKED_BY_WALKER
                return

        # 2.2: BIke avoidancd behaviors
        bike_state, bike = CarlaDataProvider.is_bike_hazard(self._hero_vehicle)
        if bike_state:
            self.agent_state = AgentState.BLOCKED_BY_BIKE

        # 2.3: Car changelane behaviors
        lane_vehicle_state, vehicle = CarlaDataProvider.is_lane_vehicle_hazard(
            self._hero_vehicle, self.target_road_option
        )
        if lane_vehicle_state:
            self.agent_state = AgentState.BLOCKED_BY_VEHICLE

        # 2.4: Car in junction  behaviors
        junction_vehicle_state, vehicle = CarlaDataProvider.is_junction_vehicle_hazard(
            self._hero_vehicle, self.target_road_option
        )
        if junction_vehicle_state:
            self.agent_state = AgentState.BLOCKED_BY_VEHICLE

        # 2.5: Car following behaviors
        vehicle_state, vehicle, distance = self._collision_and_car_avoid_manager()

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                self._hero_vehicle.bounding_box.extent.y, self._hero_vehicle.bounding_box.extent.x
            )

            # Emergency brake if the car is very close.
            if distance < self.behavior.braking_distance:
                self.agent_state = AgentState.BLOCKED_BY_VEHICLE
            else:
                vehicle_speed = CarlaDataProvider.get_speed(vehicle) * 3.6
                delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
                ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

                # Under safety time distance, slow down.
                if self.behavior.safety_time > ttc > 0.0:
                    self.speed_limit = min(
                        positive(vehicle_speed - self.behavior.speed_decrease),
                        min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)
                    )

                # Actual safety distance area, try to follow the speed of the vehicle in front.
                elif 2 * self.behavior.safety_time > ttc >= self.behavior.safety_time:

                    self.speed_limit = min(
                        max(self._min_speed, vehicle_speed),
                        min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)
                    )

                # Normal behavior.
                else:
                    self.speed_limit = min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)

        # 4: Intersection behavior
        # Checking if there's a junction nearby to slow down
        elif incoming_waypoint.is_junction and (incoming_direction == RoadOption.LEFT
                                                or incoming_direction == RoadOption.RIGHT):
            self.speed_limit = min(self.behavior.max_speed, self.speed_limit - 5)

        # 5: Change lane behavior
        elif incoming_waypoint.lane_id != self.current_waypoint.lane_id \
                or (incoming_direction == RoadOption.CHANGELANELEFT
                    or incoming_direction == RoadOption.CHANGELANERIGHT):
            self.speed_limit = max(
                min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist) * 0.6, 25
            )

        # 6: Normal behavior
        else:
            self.speed_limit = min(self.behavior.max_speed, self.speed_limit - self.behavior.speed_lim_dist)

    def _traffic_light_manager(self) -> int:
        """
        This method is in charge of behaviors for red lights and stops.

        WARNING: What follows is a proxy to avoid having a car brake after running a yellow light.
        This happens because the car is still under the influence of the semaphore,
        even after passing it. So, the semaphore id is temporarely saved to
        ignore it and go around this issue, until the car is near a new one.
        """

        light_id = self._hero_vehicle.get_traffic_light(
        ).id if self._hero_vehicle.get_traffic_light() is not None else -1
        light_state = str(self._hero_vehicle.get_traffic_light_state())

        if light_state == "Red":
            if not self.current_waypoint.is_junction and (self._light_id_to_ignore != light_id or light_id == -1):
                return 1
            elif self.current_waypoint.is_junction and light_id != -1:
                self.light_id_to_ignore = light_id
        if self._light_id_to_ignore != light_id:
            self._light_id_to_ignore = -1
        return 0

    def _pedestrian_avoid_manager(self) -> Tuple[bool, carla.Actor, float]:
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")

        def dist(w):
            return w.get_location().distance(self.current_waypoint.transform.location)

        walker_list = [w for w in walker_list if dist(w) < 10]

        if self.target_road_option == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._behavior_is_vehicle_hazard(
                walker_list,
                max(self.behavior.min_proximity_threshold, self.speed_limit / 2),
                up_angle_th=90,
                lane_offset=-1
            )
        elif self.target_road_option == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._behavior_is_vehicle_hazard(
                walker_list,
                max(self.behavior.min_proximity_threshold, self.speed_limit / 2),
                up_angle_th=90,
                lane_offset=1
            )
        else:
            walker_state, walker, distance = self._behavior_is_vehicle_hazard(
                walker_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 3), up_angle_th=60
            )

        return walker_state, walker, distance

    def _collision_and_car_avoid_manager(self) -> Tuple[bool, carla.Actor, float]:
        """
        This module is in charge of warning in case of a collision
        and managing possible overtaking or tailgating chances.

            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v):
            return v.get_location().distance(self.current_waypoint.transform.location)

        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._hero_vehicle.id]

        if self.target_road_option == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._behavior_is_vehicle_hazard(
                vehicle_list,
                max(self.behavior.min_proximity_threshold, self.speed_limit / 2),
                up_angle_th=180,
                lane_offset=-1
            )
        elif self.target_road_option == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._behavior_is_vehicle_hazard(
                vehicle_list,
                max(self.behavior.min_proximity_threshold, self.speed_limit / 2),
                up_angle_th=180,
                lane_offset=1
            )
        else:
            vehicle_state, vehicle, distance = self._behavior_is_vehicle_hazard(
                vehicle_list, max(self.behavior.min_proximity_threshold, self.speed_limit / 2), up_angle_th=30
            )

            # Check for overtaking

            if vehicle_state and self.target_road_option == RoadOption.LANEFOLLOW \
                    and not self.current_waypoint.is_junction and self._speed > 10 \
                    and self.behavior.overtake_counter == 0 \
                    and self._speed > CarlaDataProvider.get_speed(vehicle) * 3.6:
                self._overtake(vehicle_list)

            # Check for tailgating

            elif not vehicle_state and self.target_road_option == RoadOption.LANEFOLLOW \
                    and not self.current_waypoint.is_junction and self._speed > 10 \
                    and self.behavior.tailgate_counter == 0:
                self._tailgating(vehicle_list)

        return vehicle_state, vehicle, distance

    def _overtake(self, vehicle_list: List[carla.Actor]) -> None:
        """
        This method is in charge of overtaking behaviors.

            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = self.current_waypoint.left_lane_marking.lane_change
        right_turn = self.current_waypoint.right_lane_marking.lane_change

        left_wpt = self.current_waypoint.get_left_lane()
        right_wpt = self.current_waypoint.get_right_lane()

        if (left_turn == carla.LaneChange.Left or left_turn == carla.LaneChange.Both) and \
                self.current_waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
            new_vehicle_state, _, _ = self._behavior_is_vehicle_hazard(
                vehicle_list,
                max(self.behavior.min_proximity_threshold, self.speed_limit / 3),
                up_angle_th=180,
                lane_offset=-1
            )
            if not new_vehicle_state:
                _, left_end_dis = get_next_until_junction(left_wpt, 100)
                if left_end_dis > 40:
                    print("[PLANNER] Overtaking to the left!")
                    self.behavior.overtake_counter = 200
                    left_plan = generate_change_lane_route(self.current_waypoint, 'left', 0, 5, left_end_dis - 20)
                    self.change_route_in_current_road(left_plan)
        elif right_turn == carla.LaneChange.Right and self.current_waypoint.lane_id * right_wpt.lane_id > 0 and \
                right_wpt.lane_type == carla.LaneType.Driving:
            new_vehicle_state, _, _ = self._behavior_is_vehicle_hazard(
                vehicle_list,
                max(self.behavior.min_proximity_threshold, self.speed_limit / 3),
                up_angle_th=180,
                lane_offset=1
            )
            if not new_vehicle_state:
                _, right_end_dis = get_next_until_junction(right_wpt, 100)
                if right_end_dis > 40:
                    print("[PLANNER] Overtaking to the right!")
                    self.behavior.overtake_counter = 200
                    right_plan = generate_change_lane_route(self.current_waypoint, 'right', 0, 5, right_end_dis - 20)
                    self.change_route_in_current_road(right_plan)

    def _tailgating(self, vehicle_list: List[carla.Actor]) -> None:
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = self.current_waypoint.left_lane_marking.lane_change
        right_turn = self.current_waypoint.right_lane_marking.lane_change

        left_wpt = self.current_waypoint.get_left_lane()
        right_wpt = self.current_waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._behavior_is_vehicle_hazard(
            vehicle_list,
            max(self.behavior.min_proximity_threshold, self.speed_limit / 2),
            up_angle_th=180,
            low_angle_th=160
        )
        if behind_vehicle_state and self._speed < CarlaDataProvider.get_speed(behind_vehicle) * 3.6:
            if (right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and \
                    self.current_waypoint.lane_id * right_wpt.lane_id > 0 and \
                    right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._behavior_is_vehicle_hazard(
                    vehicle_list,
                    max(self.behavior.min_proximity_threshold, self.speed_limit / 2),
                    up_angle_th=180,
                    lane_offset=1
                )
                if not new_vehicle_state:
                    print("[PLANNER] Tailgating, moving to the right!")
                    self.behavior.tailgate_counter = 200
                    self.set_destination(right_wpt.transform.location, self.end_waypoint.transform.location, clean=True)
            elif left_turn == carla.LaneChange.Left and self.current_waypoint.lane_id * left_wpt.lane_id > 0 and \
                    left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._behavior_is_vehicle_hazard(
                    self.current_waypoint,
                    self._vehicle_location,
                    vehicle_list,
                    max(self.behavior.min_proximity_threshold, self.speed_limit / 2),
                    up_angle_th=180,
                    lane_offset=-1
                )
                if not new_vehicle_state:
                    print("[PLANNER] Tailgating, moving to the left!")
                    self.behavior.tailgate_counter = 200
                    self.set_destination(left_wpt.transform.location, self.end_waypoint.transform.location, clean=True)
