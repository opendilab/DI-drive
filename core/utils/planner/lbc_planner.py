import carla
from collections import deque
import numpy as np
from typing import Any, Dict

from .basic_planner import AgentState, BasicPlanner
from core.utils.simulator_utils.carla_agents.navigation import RoadOption
from core.simulators.carla_data_provider import CarlaDataProvider
from core.utils.simulator_utils.carla_agents.tools.misc import draw_waypoints


class LBCPlannerNew(BasicPlanner):

    config = dict(
        min_distance=5.0,
        resolution=15,
        threshold_before=2.5,
        threshold_after=5.0,
        fps=10,
        debug=False,
    )

    def __init__(self, cfg: Dict) -> None:
        super().__init__(cfg)

        # Max skip avoids misplanning when route includes both lanes.
        self._threshold_before = self._cfg.threshold_before
        self._threshold_after = self._cfg.threshold_after

    def run_step(self):
        assert self._route is not None

        vehicle_transform = CarlaDataProvider.get_transform(self._hero_vehicle)
        self.current_waypoint = self._map.get_waypoint(
            vehicle_transform.location, lane_type=carla.LaneType.Driving, project_to_road=True
        )

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

        u = vehicle_transform.location
        max_index = -1
        for i, (node, command) in enumerate(self._waypoints_buffer):

            v = node.transform.location
            distance = np.sqrt((u.x - v.x) ** 2 + (u.y - v.y) ** 2)

            if self.node_road_option.value == 4 and command.value != 4:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            if distance < threshold:
                max_index = i

        if max_index >= 0:
            for i in range(max_index + 1):
                self.node_waypoint, self.node_road_option = self._waypoints_buffer.popleft()
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                if self.distances:
                    self.distance_to_goal -= self.distances.popleft()

        if self._waypoints_buffer:
            self.target_waypoint, self.target_road_option = self._waypoints_buffer[0]
        self.speed_limit = self._hero_vehicle.get_speed_limit()
        self.agent_state = AgentState.NAVIGATING
        if self._debug:
            draw_waypoints(self._world, self.target_waypoint)
