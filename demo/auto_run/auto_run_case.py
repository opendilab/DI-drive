import os
import argparse
from argparse import RawTextHelpFormatter
from easydict import EasyDict

from core.envs import CarlaEnvWrapper, ScenarioCarlaEnv
from core.policy import AutoPIDPolicy
from ding.utils import set_pkg_seed
from core.simulators.srunner.tools.route_parser import RouteParser
from core.simulators.srunner.tools.scenario_parser import ScenarioConfigurationParser

casezoo_config = dict(
    env=dict(
        simulator=dict(
            planner=dict(type='behavior', ),
            n_vehicles=20,
            #n_pedestrians=25,
            disable_two_wheels=True,
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[400, 400],
                    position=[-5.5, 0, 2.8],
                    rotation=[-15, 0, 0],
                ),
                dict(
                    name='birdview',
                    type='bev',
                    size=[320, 320],
                    pixels_per_meter=6,
                ),
            ),
            waypoint_num=50,
            #debug=True,
        ),
        #no_rendering=True,
        visualize=dict(
            type='rgb',
            outputs=['show']
        ),
    ),
    policy=dict(target_speed=40, ),
)

main_config = EasyDict(casezoo_config)


def main(args, cfg, seed=0):
    configs = []
    if args.route is not None:
        routes = args.route[0]
        scenario_file = args.route[1]
        single_route = None
        if len(args.route) > 2:
            single_route = args.route[2]

        configs += RouteParser.parse_routes_file(routes, scenario_file, single_route)

    if args.scenario is not None:
        configs += ScenarioConfigurationParser.parse_scenario_configuration(args.scenario)

    carla_env = CarlaEnvWrapper(ScenarioCarlaEnv(cfg.env, args.host, args.port))
    carla_env.seed(seed)
    set_pkg_seed(seed)
    auto_policy = AutoPIDPolicy(cfg.policy).eval_mode

    for config in configs:
        auto_policy.reset([0])
        obs = carla_env.reset(config)
        while True:
            actions = auto_policy.forward({0: obs})
            action = actions[0]['action']
            timestep = carla_env.step(action)
            obs = timestep.obs
            if timestep.info.get('abnormal', False):
                # If there is an abnormal timestep, reset all the related variables(including this env).
                auto_policy.reset([0])
                obs = carla_env.reset(config)
            carla_env.render()
            if timestep.done:
                break
    carla_env.close()


if __name__ == "__main__":
    description = ("DI-drive CaseZoo Environment")

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--route', help='Run a route as a scenario (input:(route_file,scenario_file,[route id]))', nargs='+', type=str)
    parser.add_argument('--scenario', help='Run a single scenario (input: scenario name)', type=str)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=9000,
                        help='TCP port to listen to (default: 9000)', type=int)
    parser.add_argument('--tm-port', default=None,
                        help='Port to use for the TrafficManager (default: None)', type=int)

    args = parser.parse_args()

    main(args, main_config)
