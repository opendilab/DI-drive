import numpy as np
from enum import Enum
from collections import deque
from easydict import EasyDict
from typing import Dict, List, Tuple, Union
import copy
import carla

from core.utils.simulator_utils.carla_agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from core.utils.simulator_utils.carla_agents.navigation.global_route_planner import GlobalRoutePlanner
from core.utils.simulator_utils.carla_agents.navigation import RoadOption
from core.simulators.carla_data_provider import CarlaDataProvider
from core.utils.simulator_utils.carla_agents.tools.misc import draw_waypoints
from core.utils.others.config_helper import deep_merge_dicts


class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    VOID = -1
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_BY_WALKER = 3
    BLOCKED_RED_LIGHT = 4
    BLOCKED_BY_BIKE = 5


class BasicPlanner(object):
    """
    Basic local planner for Carla simulator. It can set route with a pair of start and end waypoints,
    or directly set with a waypoint list. The planner will provide target waypoint and road option
    in current route position, and record current route distance and end timeout. The planner will
    also judge agent state by checking surrounding vehicles, walkers and traffic lights.

    The route's element consists of a waypoint and a road option. Local planner uses a waypoint queue
    to store all the unreached waypoints, and a waypoint buffer to store some of the near waypoints to
    speed up searching. In short, `node` waypoint is the waypoint in route that farthest from hero
    vehicle and within ``min_distance``, and `target` waypoint is the next waypoint of node waypoint.

    :Arguments:
        - cfg (Dict): Config dict.

    :Interfaces: set_destination, set_route, run_step, get_waypoints_list, clean_up
    """

    config = dict(
        min_distance=5.0,
        resolution=5.0,
        fps=10,
        debug=False,
    )

    def __init__(self, cfg: Dict) -> None:
        if 'cfg_type' not in cfg:
            self._cfg = self.__class__.default_config()
            self._cfg = deep_merge_dicts(self._cfg, cfg)
        else:
            self._cfg = cfg
        self._hero_vehicle = CarlaDataProvider.get_hero_actor()
        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map()

        self._resolution = self._cfg.resolution
        self._min_distance = self._cfg.min_distance
        self._fps = self._cfg.fps

        self._route = None
        self._waypoints_queue = deque()
        self._buffer_size = 100
        self._waypoints_buffer = deque(maxlen=100)
        self._end_location = None

        self.current_waypoint = None
        self.node_waypoint = None
        self.target_waypoint = None
        self.node_road_option = None
        self.target_road_option = None
        self.agent_state = None
        self.speed_limit = 0

        self.distance_to_goal = 0.0
        self.distances = deque()
        self.timeout = -1
        self.timeout_in_seconds = 0

        self._debug = self._cfg.debug

        self._grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(self._map, self._resolution))
        self._grp.setup()

    def set_destination(
            self, start_location: carla.Location, end_location: carla.Location, clean: bool = False
    ) -> None:
        """
        This method creates a route of a list of waypoints from start location to destination location
        based on the route traced by the global router. If ``clean`` is set true, it will clean current
        route and waypoint queue.

        :Arguments:
            - start_location (carla.Location): initial position.
            - end_location (carla.Location): final position.
            - clean (bool): Whether to clean current route. Defaults to False.
        """
        start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)
        new_route = self._grp.trace_route(start_waypoint.transform.location, self.end_waypoint.transform.location)
        if clean:
            self._waypoints_queue.clear()
            self._waypoints_buffer.clear()
            self._route = new_route
            self.distance_to_goal = 0
            self.distances.clear()
        else:
            self._route += new_route
        CarlaDataProvider.set_hero_vehicle_route(self._route)

        prev_loc = None
        for elem in new_route:
            self._waypoints_queue.append(elem)
            cur_loc = elem[0].transform.location
            if prev_loc is not None:
                delta = cur_loc.distance(prev_loc)
                self.distance_to_goal += delta
                self.distances.append(delta)
            prev_loc = cur_loc

        self._buffer_size = min(int(100 // self._resolution), 100)
        self.node_waypoint = start_waypoint
        self.node_road_option = RoadOption.LANEFOLLOW
        self.timeout_in_seconds = ((self.distance_to_goal / 1000.0) / 5.0) * 3600.0 + 20.0
        self.timeout = self.timeout_in_seconds * self._fps

    def set_route(self, route: List, clean: bool = False) -> None:
        """
        This method add a route into planner to trace. If ``clean`` is set true, it will clean current
        route and waypoint queue.

        :Arguments:
            - route (List): Route add to planner.
            - clean (bool, optional): Whether to clean current route. Defaults to False.
        """
        if clean:
            self._waypoints_queue.clear()
            self._waypoints_buffer.clear()
            self._route = route
            self.distance_to_goal = 0
            self.distances.clear()
        else:
            self._route.extend(route)

        self.end_waypoint = self._route[-1][0]

        CarlaDataProvider.set_hero_vehicle_route(self._route)

        prev_loc = None
        for elem in route:
            self._waypoints_queue.append(elem)
            cur_loc = elem[0].transform.location
            if prev_loc is not None:
                delta = cur_loc.distance(prev_loc)
                self.distance_to_goal += delta
                self.distances.append(delta)
            prev_loc = cur_loc

        if self.distances:
            cur_resolution = np.average(list(self.distances)[:100])
            self._buffer_size = min(100, int(100 // cur_resolution))
        self.node_waypoint, self.node_road_option = self._waypoints_queue[0]
        self.timeout_in_seconds = ((self.distance_to_goal / 1000.0) / 5.0) * 3600.0 + 20.0
        self.timeout = self.timeout_in_seconds * self._fps

    def add_route_in_front(self, route):
        if self._waypoints_buffer:
            prev_loc = self._waypoints_buffer[0][0].transform.location
        else:
            prev_loc = self._waypoints_queue[0][0].transform.location
        for elem in route[::-1]:
            self._waypoints_buffer.appendleft(elem)
            cur_loc = elem[0].transform.location
            delta = cur_loc.distance(prev_loc)
            self.distance_to_goal += delta
            self.distances.appendleft(delta)
            prev_loc = cur_loc

        if len(self._waypoints_buffer) > self._buffer_size:
            for i in range(len(self._waypoints_buffer) - self._buffer_size):
                elem = self._waypoints_buffer.pop()
                self._waypoints_queue.appendleft(elem)
        self.node_waypoint, self.node_road_option = self._waypoints_buffer[0]

    def run_step(self) -> None:
        """
        Run one step of local planner. It will update node and target waypoint and road option, and check agent
        states.
        """
        assert self._route is not None

        vehicle_transform = CarlaDataProvider.get_transform(self._hero_vehicle)
        self.current_waypoint = self._map.get_waypoint(
            vehicle_transform.location, lane_type=carla.LaneType.Driving, project_to_road=True
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
        self.speed_limit = self._hero_vehicle.get_speed_limit()
        self.agent_state = AgentState.NAVIGATING

        # Detect vehicle and light hazard
        vehicle_state, vehicle = CarlaDataProvider.is_vehicle_hazard(self._hero_vehicle)
        if not vehicle_state:
            vehicle_state, vehicle = CarlaDataProvider.is_lane_vehicle_hazard(
                self._hero_vehicle, self.target_road_option
            )
        if not vehicle_state:
            vehicle_state, vehicle = CarlaDataProvider.is_junction_vehicle_hazard(
                self._hero_vehicle, self.target_road_option
            )
        if vehicle_state:
            if self._debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self.agent_state = AgentState.BLOCKED_BY_VEHICLE

        bike_state, bike = CarlaDataProvider.is_bike_hazard(self._hero_vehicle)
        if bike_state:
            if self._debug:
                print('!!! BIKE BLOCKING AHEAD [{}])'.format(bike.id))

            self.agent_state = AgentState.BLOCKED_BY_BIKE

        walker_state, walker = CarlaDataProvider.is_walker_hazard(self._hero_vehicle)
        if walker_state:
            if self._debug:
                print('!!! WALKER BLOCKING AHEAD [{}])'.format(walker.id))

            self.agent_state = AgentState.BLOCKED_BY_WALKER

        light_state, traffic_light = CarlaDataProvider.is_light_red(self._hero_vehicle)

        if light_state:
            if self._debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self.agent_state = AgentState.BLOCKED_RED_LIGHT

        if self._debug:
            draw_waypoints(self._world, self.current_waypoint)

    def get_waypoints_list(self, waypoint_num: int) -> List[carla.Waypoint]:
        """
        Return a list of wapoints from the end of waypoint buffer.

        :Arguments:
            - waypoint_num (int): Num of waypoint in list.

        :Returns:
            List[carla.Waypoint]: List of waypoint.
        """
        num = 0
        i = 0
        waypoint_list = []
        while num < waypoint_num and i < len(self._waypoints_buffer):
            waypoint = self._waypoints_buffer[i][0]
            i += 1
            if len(waypoint_list) == 0:
                waypoint_list.append(waypoint)
                num + 1
            elif waypoint_list[-1].transform.location.distance(waypoint.transform.location) > 1e-4:
                waypoint_list.append(waypoint)
                num += 1
        return waypoint_list

    def get_direction_list(self, waypoint_num: int) -> List[RoadOption]:
        num = min(waypoint_num, len(self._waypoints_buffer))
        direction_list = []
        for i in range(num):
            direction = self._waypoints_buffer[i][1].value
            direction_list.append(direction)
        return direction_list

    def get_incoming_waypoint_and_direction(self, steps: int = 3) -> Tuple[carla.Waypoint, RoadOption]:
        """
        Returns direction and waypoint at a distance ahead defined by the user.

        :Arguments:
            - steps (int): Number of steps to get the incoming waypoint.

        :Returns:
            Tuple[carla.Waypoint, RoadOption]: Waypoint and road option ahead.
        """
        if len(self._waypoints_buffer) > steps:
            return self._waypoints_buffer[steps]
        elif (self._waypoints_buffer):
            return self._waypoints_buffer[-1]
        else:
            return self.current_waypoint, RoadOption.VOID

    def clean_up(self) -> None:
        """
        Clear route, waypoint queue and buffer.
        """
        self._waypoints_queue.clear()
        self._waypoints_buffer.clear()
        if self._route is not None:
            self._route.clear()
        self.distances.clear()

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(cls.config)
        cfg.cfg_type = cls.__name__ + 'Config'
        return copy.deepcopy(cfg)
