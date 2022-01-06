'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''
from gym.envs.registration import register, registry
from core import SIMULATORS
from .base_drive_env import BaseDriveEnv
from .drive_env_wrapper import DriveEnvWrapper, BenchmarkEnvWrapper

envs = []
env_map = {}

if 'carla' in SIMULATORS:
    from .simple_carla_env import SimpleCarlaEnv
    from .scenario_carla_env import ScenarioCarlaEnv
    env_map.update({
        "SimpleCarla-v1": 'core.envs.simple_carla_env.SimpleCarlaEnv',
        "ScenarioCarla-v1": 'core.envs.scenario_carla_env.ScenarioCarlaEnv'
    })

if 'metadrive' in SIMULATORS:
    from .md_macro_env import MetaDriveMacroEnv
    env_map.update({
        "Macro-v1": 'core.envs.md_macro_env:MetaDriveMacroEnv',
    })

for k, v in env_map.items():
    if k not in registry.env_specs:
        envs.append(k)
        register(id=k, entry_point=v)

if len(envs) > 0:
    print("[ENV] Register environments: {}.".format(envs))
