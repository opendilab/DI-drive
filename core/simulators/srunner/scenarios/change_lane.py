import random
import py_trees
import carla

from core.simulators.carla_data_provider import CarlaDataProvider
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter, ActorDestroy, StopVehicle, LaneChange, WaypointFollower, Idle
)
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import \
    DriveDistance, InTriggerDistanceToVehicle, StandStill, WaitUntilInFront
from core.simulators.srunner.scenarios.basic_scenario import BasicScenario
from core.simulators.srunner.tools.scenario_helper import get_waypoint_in_distance


class ChangeLane(BasicScenario):

    def __init__(
        self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60
    ):
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        self._fast_vehicle_velocity = random.randint(8, 12)
        self._slow_vehicle_velocity = 0
        self._change_lane_velocity = 8

        self._slow_vehicle_distance = random.randint(70, 80)
        self._fast_vehicle_distance = random.randint(50, 60)
        self._trigger_distance = random.randint(45, 50)
        self._max_brake = 1

        self.direction = 'left'  # direction of lane change
        self.lane_check = 'true'  # check whether a lane change is possible

        super(ChangeLane, self).__init__(
            "ChangeLane", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable
        )

    def _initialize_actors(self, config):
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform, disable_two_wheels=True)
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)
        # fast vehicle
        # transform visible
        fast_car_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._fast_vehicle_distance)
        self.fast_car_visible = carla.Transform(
            carla.Location(
                fast_car_waypoint.transform.location.x, fast_car_waypoint.transform.location.y,
                fast_car_waypoint.transform.location.z
            ), fast_car_waypoint.transform.rotation
        )

        # slow vehicle
        # transform visible
        slow_car_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._slow_vehicle_distance)
        self.slow_car_visible = carla.Transform(
            carla.Location(
                slow_car_waypoint.transform.location.x, slow_car_waypoint.transform.location.y,
                slow_car_waypoint.transform.location.z
            ), slow_car_waypoint.transform.rotation
        )

    def _create_behavior(self):
        """
        Order of sequence:
        - sequence_slow: slow vehicle brake and stop
        - sequence_fast: fast vehicle drive for a defined distance
        - endcondition: drive for a defined distance
        """

        sequence_slow = py_trees.composites.Sequence("Slow Vehicle")
        slow_visible = ActorTransformSetter(self.other_actors[1], self.slow_car_visible)
        sequence_slow.add_child(slow_visible)
        brake = StopVehicle(self.other_actors[1], self._max_brake)
        sequence_slow.add_child(brake)
        sequence_slow.add_child(Idle())

        sequence_fast = py_trees.composites.Sequence("Fast Vehicle")
        fast_visible = ActorTransformSetter(self.other_actors[0], self.fast_car_visible)
        sequence_fast.add_child(fast_visible)
        just_drive = py_trees.composites.Parallel(
            "DrivingTowardsSlowVehicle", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        driving_fast = WaypointFollower(self.other_actors[0], self._fast_vehicle_velocity)
        just_drive.add_child(driving_fast)
        distance_to_vehicle = InTriggerDistanceToVehicle(
            self.other_actors[1], self.other_actors[0], self._trigger_distance
        )
        just_drive.add_child(distance_to_vehicle)
        sequence_fast.add_child(just_drive)

        # change lane
        lane_change = LaneChange(
            self.other_actors[0], self._change_lane_velocity, distance_lane_change=10, distance_other_lane=30
        )
        sequence_fast.add_child(lane_change)
        fast_final_drive = WaypointFollower(self.other_actors[0], self._fast_vehicle_velocity, avoid_collision=True)
        sequence_fast.add_child(fast_final_drive)

        # ego vehicle
        # end condition
        endcondition = py_trees.composites.Sequence(
            "End Condition", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        endcondition_part1 = WaitUntilInFront(self.ego_vehicles[0], self.other_actors[1])
        endcondition_part2 = DriveDistance(self.ego_vehicles[0], 30)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # build tree
        behavior = py_trees.composites.Parallel(
            "Parallel Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        behavior.add_child(sequence_slow)
        behavior.add_child(sequence_fast)
        behavior.add_child(endcondition)

        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(behavior)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(ActorDestroy(self.other_actors[1]))
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
