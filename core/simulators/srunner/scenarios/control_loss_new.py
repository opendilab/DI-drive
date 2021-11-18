import random
import py_trees
import carla

from core.simulators.carla_data_provider import CarlaDataProvider
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_behaviors import ChangeNoiseParameters, \
    ActorTransformSetter
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToLocation, InTriggerDistanceToNextIntersection, DriveDistance
)
from core.simulators.srunner.scenarios.basic_scenario import BasicScenario
from core.simulators.srunner.tools.scenario_helper import get_location_in_distance_from_wp


class ControlLossNew(BasicScenario):

    def __init__(
        self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60
    ):
        self._no_of_jitter = 10
        self._noise_mean = 0
        self._noise_std = 0.01
        self._dynamic_mean_for_steer = 0.001
        self._dynamic_mean_for_throttle = 0.045
        self._abort_distance_to_intersection = 10
        self._current_steer_noise = [0]  # This is a list, since lists are mutable
        self._current_throttle_noise = [0]
        self._start_distance = 20
        self._trigger_dist = 2
        self._end_distance = 30
        self._ego_vehicle_max_steer = 0.0
        self._ego_vehicle_max_throttle = 1.0
        self._ego_vehicle_target_velocity = 15
        self._map = CarlaDataProvider.get_map()
        # Timeout of scenario in seconds
        self.timeout = timeout
        # The reference trigger for the control loss
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self.object = []
        super(ControlLossNew, self).__init__(
            "ControlLossNew", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable
        )

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._distance = random.sample(range(10, 80), 3)
        self._distance = sorted(self._distance)
        self._dist_prop = [x - 2 for x in self._distance]

        self.first_location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._dist_prop[0])
        self.second_location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._dist_prop[1])
        self.third_location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._dist_prop[2])

        self.first_transform = carla.Transform(self.first_location)
        self.second_transform = carla.Transform(self.second_location)
        self.third_transform = carla.Transform(self.third_location)
        self.first_transform = carla.Transform(
            carla.Location(self.first_location.x, self.first_location.y, self.first_location.z)
        )
        self.second_transform = carla.Transform(
            carla.Location(self.second_location.x, self.second_location.y, self.second_location.z)
        )
        self.third_transform = carla.Transform(
            carla.Location(self.third_location.x, self.third_location.y, self.third_location.z)
        )

        first_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self.first_transform)
        second_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self.second_transform)
        third_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self.third_transform)

        first_debris.set_transform(self.first_transform)
        second_debris.set_transform(self.second_transform)
        third_debris.set_transform(self.third_transform)

        self.object.extend([first_debris, second_debris, third_debris])
        for debris in self.object:
            debris.set_simulate_physics(False)

        self.other_actors.append(first_debris)
        self.other_actors.append(second_debris)
        self.other_actors.append(third_debris)

    def _create_behavior(self):

        # start condition
        start_end_parallel = py_trees.composites.Parallel(
            "Jitter", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        start_condition = InTriggerDistanceToLocation(self.ego_vehicles[0], self.first_location, self._trigger_dist)
        for _ in range(self._no_of_jitter):

            # change the current noise to be applied
            turn = ChangeNoiseParameters(
                self._current_steer_noise, self._current_throttle_noise, self._noise_mean, self._noise_std,
                self._dynamic_mean_for_steer, self._dynamic_mean_for_throttle
            )  # Mean value of steering noise
        # Noise end! put again the added noise to zero.
        noise_end = ChangeNoiseParameters(self._current_steer_noise, self._current_throttle_noise, 0, 0, 0, 0)

        jitter_action = py_trees.composites.Parallel("Jitter", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # Abort jitter_sequence, if the vehicle is approaching an intersection
        jitter_abort = InTriggerDistanceToNextIntersection(self.ego_vehicles[0], self._abort_distance_to_intersection)
        # endcondition: Check if vehicle reached waypoint _end_distance from here:
        end_condition = DriveDistance(self.ego_vehicles[0], self._end_distance)
        start_end_parallel.add_child(start_condition)
        start_end_parallel.add_child(end_condition)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("ControlLoss")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self.first_transform, physics=False))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self.second_transform, physics=False))
        sequence.add_child(ActorTransformSetter(self.other_actors[2], self.third_transform, physics=False))
        jitter = py_trees.composites.Sequence("Jitter Behavior")
        jitter.add_child(turn)
        jitter.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], self.second_location, self._trigger_dist))
        jitter.add_child(turn)
        jitter.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], self.third_location, self._trigger_dist))
        jitter.add_child(turn)
        jitter_action.add_child(jitter)
        jitter_action.add_child(jitter_abort)
        sequence.add_child(start_end_parallel)
        sequence.add_child(jitter_action)
        sequence.add_child(end_condition)
        sequence.add_child(noise_end)
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

    def change_control(self, control):
        """
        This is a function that changes the control based on the scenario determination
        :param control: a carla vehicle control
        :return: a control to be changed by the scenario.
        """
        control.steer += self._current_steer_noise[0]
        control.throttle += self._current_throttle_noise[0]

        return control

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
