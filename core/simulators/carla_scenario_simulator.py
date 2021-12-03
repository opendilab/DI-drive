import os
from typing import Any, Dict, List, Optional

import carla

from core.simulators.carla_simulator import CarlaSimulator
from core.simulators.carla_data_provider import CarlaDataProvider

from .srunner.scenarios.route_scenario import RouteScenario, SCENARIO_CLASS_DICT
from .srunner.scenariomanager.scenario_manager import ScenarioManager


class CarlaScenarioSimulator(CarlaSimulator):
    """
    Simulator used to run scenarios in Carla.
    The simulator loads the config instance of the provided scenario to start simulation.
    It can create hero actor, NPC vehicles, walkers, world map according to it and the
    configuration dict. The modification of sensors and planners, and the running status
    and information you can get from it are set the same as ``CarlaSimulator``.

    When created, it will set up Carla client and store default configuration the same as
    ``CarlaSimulator``, and it can also be change by the input arguments of the ``init``
    method.

    If no traffic manager port is provided, it will find random free port in system.

    :Arguments:
        - cfg (Dict): Config Dict.
        - client (carla.Client, optional): Already established Carla client. Defaults to None.
        - host (str, optional): TCP host Carla client link to. Defaults to 'localhost'.
        - port (int, optional): TCP port Carla client link to. Defaults to 9000.
        - tm_port (int, optional): Traffic manager port Carla client link to. Defaults to None.
        - timeout (float, optional): Carla client link timeout. Defaults to 10.0.

    :Interfaces:
        init, get_state, get_sensor_data, get_navigation, get_information, apply_control, run_step, clean_up

    :Properties:
        - town_name (str): Current town name.
        - hero_player (carla.Actor): hero actor in simulation.
        - collided (bool): Whether collided in current episode.
        - end_distance (float): Distance to target in current frame.
        - end_timeout (float): Timeout for entire route provided by planner.
        - total_distance (float): Distance for entire route provided by planner.
        - scenario_manager (Any): Scenario Manager instance used to get running state.
    """

    config = dict(
        town='Town01',
        weather='random',
        sync_mode=True,
        delta_seconds=0.1,
        no_rendering=False,
        auto_pilot=False,
        n_vehicles=0,
        n_pedestrians=0,
        disable_two_wheels=False,
        col_threshold=400,
        resolution=1.0,
        waypoint_num=20,
        obs=list(),
        planner=dict(),
        aug=None,
        verbose=True,
        debug=False,
    )

    def __init__(
            self,
            cfg: Dict,
            client: Optional[carla.Client] = None,
            host: str = 'localhost',
            port: int = 9000,
            tm_port: int = 9050,
            timeout: float = 10.0,
            **kwargs
    ) -> None:
        """
        Init Carla scenario simulator.
        """
        super().__init__(cfg, client, host, port, tm_port, timeout)
        self._resolution = self._cfg.resolution
        self._scenario = None
        self._start_scenario = False
        self._manager = ScenarioManager(self._debug, self._sync_mode, self._client_timeout)
        self._criteria_status = dict()

    def init(self, config: Any) -> None:
        """
        Init simulator episode with provided args.
        This method takes an scenario configuration instance to set up scenarios in Carla server. the scenario could be
        a single scenario, or a route scenario together with several scenarios during navigating the route. A scenario
        manager is used to manager and check the running status and tick scenarios. A local planner is set to trace the
        route to generate target waypoint and road options in each tick. It will set world, map, vehicles, pedestrians
        dut to provided args and default configs, and reset running status. If no collision happens when creating
        actors, the init will end and return.

        :Arguments:
            - config (Any): Scenario configuration instance, containing information about the scenarios.
        """
        self._scenario_config = config
        self.clean_up()
        self._set_town(config.town)
        self._set_weather(self._weather)

        self._blueprints = self._world.get_blueprint_library()

        while True:
            self.clean_up()

            CarlaDataProvider.set_client(self._client)
            CarlaDataProvider.set_world(self._world)
            CarlaDataProvider.set_traffic_manager_port(self._tm.get_port())

            if CarlaDataProvider.get_map().name != config.town and CarlaDataProvider.get_map().name != "OpenDriveMap":
                print("WARNING: The CARLA server uses the wrong map: {}".format(CarlaDataProvider.get_map().name))
                print("WARNING: This scenario requires to use map: {}".format(config.town))

            print("[SIMULATOR] Preparing scenario: " + config.name)
            config.n_vehicles = self._n_vehicles
            config.disable_two_wheels = self._disable_two_wheels

            if "RouteScenario" in config.name:
                self._scenario = RouteScenario(
                    world=self._world, config=config, debug_mode=self._debug, resolution=self._resolution
                )
                self._hero_actor = self._scenario.ego_vehicles[0]
                self._prepare_observations()
                self._manager.load_scenario(self._scenario)
                self._planner.set_route(CarlaDataProvider.get_hero_vehicle_route(), clean=True)
                self._total_distance = self._planner.distance_to_goal
                self._end_timeout = self._scenario.route_timeout

            else:
                # select scenario
                if config.type in SCENARIO_CLASS_DICT:
                    scenario_class = SCENARIO_CLASS_DICT[config.type]
                    ego_vehicles = []
                    for vehicle in config.ego_vehicles:
                        ego_vehicles.append(
                            CarlaDataProvider.request_new_actor(
                                vehicle.model,
                                vehicle.transform,
                                vehicle.rolename,
                                True,
                                color=vehicle.color,
                                actor_category=vehicle.category
                            )
                        )
                    self._scenario = scenario_class(
                        world=self._world, ego_vehicles=ego_vehicles, config=config, debug_mode=self._debug
                    )
                else:
                    raise RuntimeError("Scenario '{}' not support!".format(config.type))
                self._hero_actor = self._scenario.ego_vehicles[0]
                self._prepare_observations()
                self._manager.load_scenario(self._scenario)
                self._planner.set_destination(config.route.data[0], config.route.data[1], clean=True)
                self._total_distance = self._planner.distance_to_goal

            self._spawn_pedestrians()

            if self._ready():
                if self._debug:
                    self._count_actors()
                break

    def run_step(self) -> None:
        """
        Run one step simulation.
        This will tick Carla world and scenarios, update information for all sensors and measurement.
        """
        if not self._start_scenario:
            self._manager.start_scenario()
            self._start_scenario = True

        self._tick += 1
        world_snapshot = self._world.get_snapshot()
        timestamp = world_snapshot.timestamp
        self._timestamp = timestamp.elapsed_seconds
        self._manager.tick_scenario(timestamp)

        if self._planner is not None:
            self._planner.run_step()

        self._collided = self._collision_sensor.collided
        self._traffic_light_helper.tick()

        if self._bev_wrapper is not None:
            if CarlaDataProvider._hero_vehicle_route is not None:
                self._bev_wrapper.tick()

    def get_criteria(self) -> List:
        """
        Get criteria status list of scenario in current frame. Criteria related with hero actor is encounted.

        :Returns:
            List: Criteria list of scenario.
        """
        criterion_list = self._manager.analyze_tick()
        for name, actor_id, result, actual_value, expected_value in criterion_list:
            if actor_id == self._hero_actor.id:
                self._criteria_status.update({name: [result, actual_value, expected_value]})
        return self._criteria_status

    def end_scenario(self) -> None:
        """
        End current scenario. Must be called before ending an episode.
        """
        if self._start_scenario:
            self._manager.end_scenario()
            self._start_scenario = False

    def clean_up(self) -> None:
        """
        Destroy all actors and sensors in current world. Clear all messages saved in simulator and data provider,
        and clean up running scenarios. This will NOT destroy theCarla client, so simulator can use same carla
        client to start next episode.
        """
        if self._manager is not None:
            self._manager.clean_up()
        self._criteria_status.clear()
        super().clean_up()

    @property
    def scenario_manager(self) -> Any:
        return self._manager
