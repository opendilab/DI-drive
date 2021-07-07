#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
This module provides the key configuration parameters for a route-based scenario
"""

import carla
from core.utils.planner import RoadOption

from core.simulators.srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration


class RouteConfiguration(object):
    """
    This class provides the basic  configuration for a route
    """

    def __init__(self, route=None):
        self.data = route

    def parse_xml(self, node):
        """
        Parse route config XML
        """
        self.data = []

        for waypoint in node.iter("waypoint"):
            x = float(waypoint.attrib.get('x', 0))
            y = float(waypoint.attrib.get('y', 0))
            z = float(waypoint.attrib.get('z', 0))
            c = waypoint.attrib.get('connection', '')

            self.data.append(carla.Location(x, y, z))


class RouteScenarioConfiguration(ScenarioConfiguration):
    """
    Basic configuration of a RouteScenario
    """

    trajectory = None
    scenario_file = None
