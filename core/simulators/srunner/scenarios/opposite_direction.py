import math
import py_trees
import carla
from six.moves.queue import Queue  # pylint: disable=relative-import

from core.simulators.carla_data_provider import CarlaDataProvider
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter, ActorDestroy, ActorSource, ActorSink, WaypointFollower
)
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from core.simulators.srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance
from core.simulators.srunner.scenarios.basic_scenario import BasicScenario
from core.simulators.srunner.tools.scenario_helper import get_waypoint_in_distance


class OppositeDirection(BasicScenario):
    """
    "Vehicle Maneuvering In Opposite Direction" (Traffic Scenario 05)

    This is a single ego vehicle scenario
    """

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True,
        obstacle_type='barrier',
        timeout=120
    ):
        """
        Setup all relevant parameters and create scenario
        obstacle_type -> flag to select type of leading obstacle. Values: vehicle, barrier
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._ego_vehicle_drive_distance = 100
        self._opposite_speed = 5.56  # m/s
        self._source_gap = 10  # m
        self._source_transform = None
        self._sink_location = None
        self._blackboard_queue_name = 'ManeuverOppositeDirection/actor_flow_queue'
        self._queue = py_trees.blackboard.Blackboard().set(self._blackboard_queue_name, Queue())
        self._obstacle_type = obstacle_type
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(OppositeDirection, self).__init__(
            "OppositeDirection", ego_vehicles, config, world, debug_mode, criteria_enable=criteria_enable
        )

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        other_actor_transform = config.other_actors[0].transform
        other_actor_waypoint = self._map.get_waypoint(other_actor_transform.location)
        first_vehicle_transform = carla.Transform(
            carla.Location(
                config.other_actors[0].transform.location.x, config.other_actors[0].transform.location.y,
                config.other_actors[0].transform.location.z
            ), config.other_actors[0].transform.rotation
        )
        other_actor = CarlaDataProvider.request_new_actor(config.other_actors[0].model, other_actor_transform)
        other_actor.set_transform(first_vehicle_transform)
        other_actor.set_simulate_physics(enabled=False)
        self.other_actors.append(other_actor)

        self._source_transform = other_actor_transform
        sink_waypoint = other_actor_waypoint.next(1)[0]
        while not sink_waypoint.is_intersection:
            sink_waypoint = sink_waypoint.next(1)[0]
        while sink_waypoint.is_intersection:
            sink_waypoint = sink_waypoint.next(1)[0]
        while not sink_waypoint.is_intersection:
            sink_waypoint = sink_waypoint.next(1)[0]
        self._sink_location = sink_waypoint.transform.location

        self._other_actor_transform = other_actor_transform

    def _create_behavior(self):
        """
        The behavior tree returned by this method is as follows:
        The ego vehicle is trying to pass a leading vehicle in the same lane
        by moving onto the oncoming lane while another vehicle is moving in the
        opposite direction in the oncoming lane.
        """

        # Leaf nodes
        actor_source = ActorSource(
            ['vehicle.audi.tt', 'vehicle.tesla.model3', 'vehicle.nissan.micra'], self._source_transform,
            self._source_gap, self._blackboard_queue_name
        )
        actor_sink = ActorSink(self._sink_location, 10)
        ego_drive_distance = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_drive_distance)
        waypoint_follower = WaypointFollower(
            self.other_actors[0],
            self._opposite_speed,
            blackboard_queue_name=self._blackboard_queue_name,
            avoid_collision=True
        )

        # Non-leaf nodes
        parallel_root = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # Building tree
        parallel_root.add_child(ego_drive_distance)
        parallel_root.add_child(actor_source)
        parallel_root.add_child(actor_sink)
        parallel_root.add_child(waypoint_follower)

        scenario_sequence = py_trees.composites.Sequence()
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        scenario_sequence.add_child(parallel_root)
        scenario_sequence.add_child(ActorDestroy(self.other_actors[0]))

        return scenario_sequence

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
