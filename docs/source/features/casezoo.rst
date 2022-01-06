Carla Scenario and Casezoo Evaluation
########################################

.. toctree::
    :maxdepth: 2

Carla ScenarioRunner is a module that allows traffic scenario definition and execution for the CARLA simulator.
DI-drive refers to the main definition and usage of ScenarioRunner and builds an individual scenario system --
Casezoo. In short, Casezoo is an Auto-driving scenario dataset built with Carla and run with DI-drive.
DI-drive refer to the existing Carla `scenario runner <https://github.com/carla-simulator/scenario_runner>`_
to set up scenarios, and add interfaces to suit for DI-drive, so as to run Deep Learning method easily.
DI-drive also supports both ScenarioRunner's example scenarios and routes. They can be run in the same way as
follow.


Overview
=====================

Currently Deep Learning literatures in Carla simulator mostly use **Benchmark** env setting to train and evaluate
their policies. Benchmark defines several suites of routes by a start and end point, divided into three difficulties. Each suite
specifies running parameters like weathers and amount of other vehicles and walkers in environment. The ego-vehicle
need to finish the route navigation with or without any collision. The key concept of Benchmark is that the
init position of NPCs are randomly selected, which makes each episode, even running the same route in a suite,
is different and uncontrollable. The successful rate represents some characteristics of the performance, but not
accurately reflect the ability to handle various cases.

Carla `scenario runner <https://github.com/carla-simulator/scenario_runner>`_ is a toolkit that can run scenarios
of several form. Scenario runner allows user to make and run single case with the behavior of npc and traffic lights
controllable and repeatable. However, the default cases in scenario runner is hard to interface with manually
designed driving policy, and to run a IL & RL environments. DI-drive make the scenario running very simple and
redesigned cases according to some real driving cases.

**DI-drive Casezoo** is an Auto-driving scenario datasets. It aims to establish specified scenarios to train and
evaluate with Decision Intelligence policies. The scenario defines NPC vehicles, walkers and traffic light behaviors, together
with some criteria. It succeeds only if all the behaviors operate obeying **fixed order and logic**, and may fail if
any criteria raise failure status, so that the success and failure case can exactly show the detail characteristics of the
evaluated policy. What is more, the case is able to run RL training. The reward comes from standard driving status as
well as scenario criteria. Di-drive runs Casezoo with a standard form
environment defined the same as ``gym.Env``, together with configuration in same form with other environments.
This makes the RL training quickly and conveniently for users.

In short, the new malicious introduced in DI-drive Casezoo compared with Benchmark and scenario runner includes:

- Scenario designed from real data and road test
- Scenario operating in fixed logic and order
- Suitable for IL & RL training
- Interacted with same interfaces in ``Env``


Running Guide
=================

The scenario defines a route to follow in `Town03`, `Town04` and `Town05`. The route is stored in a `.xml` file.
There are also individual scenarios defined in `python` files. The individual scenario can be triggered when hero
vehicle passes by during navigating in the route. The configuration os these scenarios in each route is stored in
a `.json` file with the same file name as the route file.

Meanwhile, you can run a single individual scenario in a certain location in town maps. They are defined in example
`xml` files. Each type of scenario possess sevaral cases at a specific location and a name saved in the `xml` file.
Details of all routes and scenarios are shown in
`Casezoo instruction <https://github.com/opendilab/DI-drive/blob/main/docs/casezoo_instruction.md>`_.

DI-drive defines :class:`ScenarioCarlaEnv <core.envs.scenario_carla_env.ScenarioCarlaEnv>`, which uses
:class:`ScenarioSimulator <core.simulators.carla_scenario_simulator.ScenarioSimulator>` to run Casezoo.
You need to parse the scenario files and get a configuration instance, then pass the configuration into
the ``reset`` method of a ``ScenarioEnv``. The environment is able to get the scenario type by itself
and create map, NPCs, behaviors and criteria in the environments. The environment can then be used as
common RL environments to train and evaluate.

Simple usage of route scenario is shown follow:

.. code:: python

    from core.simulators.srunner.tools.route_parser import RouteParser

    route_file = 'xxx.xml'
    config_file = 'xxx.json'
    config = RouteParser.parse_routes_file(routes, scenario_file)

    carla_env = ScenarioCarlaEnv(env_cfg, host, port)
    obs = carla_env.reset(config)

Simple usage of single scenario is shown follow:

.. code:: python

    from core.simulators.srunner.tools.scenario_parser import ScenarioConfigurationParser

    scenario_name = 'xxx'
    config = ScenarioConfigurationParser.parse_scenario_configuration(scenario_name)

    carla_env = ScenarioCarlaEnv(env_cfg, host, port)
    obs = carla_env.reset(config)

.. note::

    The name of single scenario (such as **CutIn**) is different from the name of a scenario case
    (such as **CutIn_1**). The former only defines the type of scenario, the latter can be run as a
    single case to start an episode in scenario environments.


You can run auto policy in scenario environments with ``demo/auto_run/auto_run_case.py``
