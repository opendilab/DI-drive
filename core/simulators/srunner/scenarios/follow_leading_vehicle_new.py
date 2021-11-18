import random

import py_trees

import carla

from core.simulators.carla_data_provider import CarlaDataProvider
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter, ActorDestroy, KeepVelocity, StopVehicle, WaypointFollower
)
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToVehicle, DriveDistance, StandStill
)
from core.simulators.srunner.scenariomanager.timer import TimeOut
from core.simulators.srunner.scenarios.basic_scenario import BasicScenario
from core.simulators.srunner.tools.scenario_helper import get_waypoint_in_distance
from core.utils.planner import RoadOption


class FollowLeadingVehicleNew(BasicScenario):

    def __init__(
        self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60
    ):
        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = random.randint(20, 30)
        self._first_vehicle_speed = random.randint(8, 12)
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._other_actor_leading_distance = 50
        self._other_actor_transform = None
        self.timeout = timeout
        self._ego_other_distance_start = random.randint(4, 8)
        super(FollowLeadingVehicleNew, self).__init__(
            "FollowLeadingVehicleNew", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable
        )

    def _initialize_actors(self, config):

        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        self._other_actor_transform = carla.Transform(
            carla.Location(
                first_vehicle_waypoint.transform.location.x, first_vehicle_waypoint.transform.location.y,
                first_vehicle_waypoint.transform.location.z
            ), first_vehicle_waypoint.transform.rotation
        )
        first_vehicle_transform = carla.Transform(
            carla.Location(
                self._other_actor_transform.location.x, self._other_actor_transform.location.y,
                self._other_actor_transform.location.z
            ), self._other_actor_transform.rotation
        )
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.nissan.patrol', first_vehicle_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):

        start_transform = ActorTransformSetter(self.other_actors[0], self._other_actor_transform)

        target_waypoint = CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location())
        # Generating waypoint list till next intersection
        plan = []
        wp_choice = target_waypoint.next(1.0)
        while True:
            target_waypoint = wp_choice[0]
            if target_waypoint.transform.location.distance(self._other_actor_transform.location
                                                           ) > self._other_actor_leading_distance:
                break
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(1.0)

        follow = py_trees.composites.Parallel('Follow Lead', policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        leading = WaypointFollower(self.other_actors[0], self._first_vehicle_speed, plan=plan)
        endcondition = DriveDistance(self.ego_vehicles[0], self._other_actor_leading_distance, name="DriveDistance")
        follow.add_child(leading)
        follow.add_child(endcondition)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(start_transform)
        sequence.add_child(follow)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
