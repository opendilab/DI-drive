#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Collection of traffic scenarios where the ego vehicle (hero)
is making a right turn
"""

from __future__ import print_function

import sys

import py_trees
import carla
from core.utils.planner import RoadOption

from core.simulators.carla_data_provider import CarlaDataProvider
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter, ActorDestroy, StopVehicle, SyncArrival, WaypointFollower, TrafficLightStateSetter
)
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import \
    DriveDistance, InTriggerDistanceToLocation
from core.simulators.srunner.scenarios.basic_scenario import BasicScenario
from core.simulators.srunner.tools.scenario_helper import (
    get_geometric_linear_intersection, get_crossing_point, generate_target_waypoint
)


class SignalizedJunctionStraight(BasicScenario):
    """
    Implementation class for Hero
    Vehicle turning right at signalized junction scenario,
    Traffic Scenario 09.

    This is a single ego vehicle scenario
    """

    def __init__(
        self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60
    ):
        """
        Setup all relevant parameters and create scenario
        """
        self._target_vel = 6.9
        self._brake_value = 0.5
        self._ego_distance = 70
        self._traffic_light = None
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout
        super(SignalizedJunctionStraight, self).__init__(
            "GoStraightAtSignalizedJunction", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable
        )

        # self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        # if self._traffic_light is None:
        #     print("No traffic light for the given location of the ego vehicle found")
        #     sys.exit(-1)
        # self._traffic_light.set_state(carla.TrafficLightState.Green)
        # self._traffic_light.set_red_time(self.timeout)
        # # other vehicle's traffic light
        # traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        # if traffic_light_other is None:
        #     print("No traffic light for the given location of the other vehicle found")
        #     sys.exit(-1)
        # traffic_light_other.set_state(carla.TrafficLightState.Green)
        # traffic_light_other.set_green_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(
                config.other_actors[0].transform.location.x, config.other_actors[0].transform.location.y,
                config.other_actors[0].transform.location.z
            ), config.other_actors[0].transform.rotation
        )
        first_vehicle = CarlaDataProvider.request_new_actor(
            config.other_actors[0].model, first_vehicle_transform, disable_two_wheels=True
        )
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        self._traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)

        if config.trigger_points is not None:
            trigger_waypoint = CarlaDataProvider.get_map().get_waypoint(config.trigger_points[0].location)
            self._traffic_light = CarlaDataProvider.get_next_traffic_light_from_waypoint(trigger_waypoint)

        # if self._traffic_light is None or self._traffic_light_other is None:
        #     raise RuntimeError("No traffic light for the given location found")

    def _create_behavior(self):
        """
        Hero vehicle is turning right in an urban area,
        at a signalized intersection, while other actor coming straight
        from left.The hero actor may turn right either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """

        set_traffic_light = py_trees.composites.Sequence("Traffic Light Setter")
        set_traffic_light = py_trees.composites.Sequence("Traffic Light Setter")
        if self._traffic_light is not None:
            set_light_green = TrafficLightStateSetter(self._traffic_light, carla.TrafficLightState.Green)
            set_traffic_light.add_child(set_light_green)
        if self._traffic_light_other is not None:
            set_other_light_green = TrafficLightStateSetter(self._traffic_light_other, carla.TrafficLightState.Green)
            set_traffic_light.add_child(set_other_light_green)

        # Selecting straight path at intersection
        target_waypoint = generate_target_waypoint(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0
        )
        # Generating waypoint list till next intersection
        plan = []
        wp_choice = target_waypoint.next(1.0)
        while not wp_choice[0].is_intersection:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(1.0)

        move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan)
        move_free = WaypointFollower(self.other_actors[0], self._target_vel)
        #stop = StopVehicle(self.other_actors[0], self._brake_value)

        # stop other actor
        move_actor_sequence = py_trees.composites.Sequence()
        move_actor_sequence.add_child(move_actor)
        move_actor_sequence.add_child(move_free)
        #move_actor_sequence.add_child(stop)
        #move_actor_sequence.add_child(ActorDestroy(self.other_actors[0]))

        # end condition
        #waypoint_follower_end = InTriggerDistanceToLocation(self.other_actors[0], plan[-1][0].transform.location, 10)
        drive = DriveDistance(self.ego_vehicles[0], self._ego_distance)
        end_condition = py_trees.composites.Parallel(
            name='End Condition', policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        #end_condition.add_child(waypoint_follower_end)
        end_condition.add_child(drive)

        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        behavior.add_child(move_actor_sequence)
        behavior.add_child(end_condition)

        sequence = py_trees.composites.Sequence()
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(set_traffic_light)
        sequence.add_child(behavior)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collison_criteria)

        return criteria

    def __del__(self):
        self._traffic_light = None
        self.remove_all_actors()
