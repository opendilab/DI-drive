'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''
from core import SIMULATORS

if 'carla' in SIMULATORS:
    from .map_utils import BeVWrapper
