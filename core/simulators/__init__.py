'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''
from core import SIMULATORS
from .fake_simulator import FakeSimulator

if 'carla' in SIMULATORS:
    from .carla_simulator import CarlaSimulator
    from .carla_scenario_simulator import CarlaScenarioSimulator
