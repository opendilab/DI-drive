import random
import py_trees
import carla

from core.simulators.carla_data_provider import CarlaDataProvider
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter, ActorDestroy, Idle, LaneChange, AccelerateToCatchUp, KeepVelocity, WaypointFollower
)
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import \
    InTriggerDistanceToVehicle, DriveDistance, WaitUntilInFront
from core.simulators.srunner.scenarios.basic_scenario import BasicScenario


class CutIn(BasicScenario):

    def __init__(
        self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60
    ):
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        self._velocity = random.randint(12, 15)
        self._delta_velocity = 10
        self._trigger_distance = random.randint(10, 40)

        self._config = config
        self._direction = None
        self._transform_visible = None

        super(CutIn, self).__init__("CutIn", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable)

    def _initialize_actors(self, config):

        # add actors from xml file
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform, disable_two_wheels=True)
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)

        # transform visible
        other_actor_transform = self.other_actors[0].get_transform()
        self._transform_visible = carla.Transform(
            carla.Location(
                other_actor_transform.location.x, other_actor_transform.location.y, other_actor_transform.location.z
            ), other_actor_transform.rotation
        )

        self._direction = config.other_actors[0].direction

    def _create_behavior(self):
        """
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - accelerate: accelerate to catch up distance to ego_vehicle
        - lane_change: change the lane
        - endcondition: drive for a defined distance
        """

        # car_visible
        cut_in = py_trees.composites.Sequence("CarOn_{}_Lane".format(self._direction))
        car_visible = ActorTransformSetter(self.other_actors[0], self._transform_visible)
        cut_in.add_child(car_visible)

        # accelerate
        accelerate = AccelerateToCatchUp(
            self.other_actors[0],
            self.ego_vehicles[0],
            throttle_value=1,
            delta_velocity=self._delta_velocity,
            trigger_distance=5,
            max_distance=500
        )
        cut_in.add_child(accelerate)

        # lane_change
        if self._direction == 'left':
            lane_change = LaneChange(
                self.other_actors[0],
                speed=self._velocity,
                direction='right',
                distance_same_lane=10,
                distance_other_lane=20
            )
            cut_in.add_child(lane_change)
        else:
            lane_change = LaneChange(
                self.other_actors[0],
                speed=self._velocity,
                direction='left',
                distance_same_lane=10,
                distance_other_lane=20
            )
            cut_in.add_child(lane_change)

        # keep velocity
        final_driving = WaypointFollower(self.other_actors[0], self._velocity, avoid_collision=True)
        cut_in.add_child(final_driving)

        # endcondition
        endcondition = py_trees.composites.Sequence(
            "End Condition", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        endcondition_part1 = WaitUntilInFront(self.other_actors[0], self.ego_vehicles[0], check_distance=False)
        endcondition_part2 = DriveDistance(self.ego_vehicles[0], 30)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # build tree
        behavior = py_trees.composites.Parallel("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        behavior.add_child(cut_in)
        behavior.add_child(endcondition)

        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(car_visible)
        sequence.add_child(behavior)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria is created, which is later used in the parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors after deletion.
        """
        self.remove_all_actors()
