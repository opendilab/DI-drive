Simulator Features
#####################

.. toctree::
    :maxdepth: 2

Overview
=============

DI-drive defines simulator instance to unify the interfaces and standardize sub-modules. As a result, users can customize
their simulator setting with a config dictionary, and get expected data by calling the instance's methods.
DI-drive currently uses Carla 0.9.9.4 for simulating Autonomous Driving. The Interaction between Carla server
and DI-drive is proceeded through :class:`CarlaSimulator <core.simulators.carla_simulator.CarlaSimulator>`.
The simulator contains a world map with roads, buildings and weather. The simulator has a hero vehicle for the user
to control and get observations. The observation may come from a **sensor** created in the simulator associated to hero
vehicle, a **planner** in the simulator to get target waypoints and select road options, and other running status returned
by methods.

Carla Simulator Configuring
=============================

All the settings of simulator can be customized by a config dict. Here 
we show the default setting of the simulator and explain its contents.

.. code:: python

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
        waypoint_num=20,
        obs=list(),
        planner=dict(),
        aug=None,
        verbose=True,
        debug=False,
    )

.. note::
    Some of the configures can be changed by provided arguments when running 
    :func:`init <core.simulators.carla_simulator.CarlaSimulator>` method, in order to make difference among episodes.

Common configurations
----------------------

Here we explain some common configurations about world map and npc settings in simulator, the instruction of 'planner', 'obs' and
'aug' are shown next.

- town
    Name of the town map used in the simulator.
- weather
    Weather setting in the simulator. List as followings. If set to 'random', the simulator will randomly choose one.
    
    .. list-table::

        *   -   Value 
            -   Weather
        *   -   1
            -   Clear Noon
        *   -   2 
            -   Cloudy Noon
        *   -   3
            -   Wet Noon
        *   -   4
            -   Wet Cloudy Noon
        *   -   5
            -   Mid Rainy Noon
        *   -   6
            -   Hard Rain Noon
        *   -   7
            -   Soft Rain Noon
        *   -   8
            -   Clear Sunset
        *   -   9
            -   Cloudy Sunset
        *   -   10
            -   Wet Sunset
        *   -   11
            -   Wet Cloudy Sunset
        *   -   12
            -   Mid Rain Sunset
        *   -   13
            -   Hard Rain Sunset
        *   -   14
            -   Soft Rain Sunset

- sync_mode
    Whether to run the simulator in synchronous mode. It is suggested to set to TRUE throughout the simulation.
- delta_seconds
    Time step in seconds between two frames. Only make sense in sync mode. We suggest to remain same throughout the simulation.
- no_rendering
    Whether to use no rendering mode in carla. If no camera sensor is needed, switching this on may speed up the simulating.
- auto_pilot
    Whether to use auto-pilot mode to control hero vehicle.
- n_vehicles
    Num of background vehicles in the world. Will be spawned at a random position.
- n_pedestrians
    Num of background pedestrians in the world. Will be spawned at a random position.
- disable_two_wheels
    Whether to disable two-wheel vehicles in NPCs.
- col_threshold
    The threshold of collision sensor intensity.
- waypoint_num
    The waypoints num ahead vehicle in waypoint_list of observations.
- verbose
    Whether print verbosely when running.

Users may need to check Carla documents for detailed information about configs.

Planner configurations
-----------------------

Here we explain the 'planner' key in config.

The Planner in Carla simulator is modified from default Carla planners. A global planner in Carla Python API is used
to generate route and road options from start to end location. The local planner is defined in DI-drive to add navigation
into observation. It provides current waypoint location, next waypoint location, road option command
and distance to final target. Planners may also take nearby walkers, vehicles and traffic lights into account.
Currently :class:`BasicPlanner <core.utils.planner.basic_planner.BasicPlanner>` and
:class:`BehaviorPlanner <core.utils.planner.behavior_planner.BehaviorPlanner>` are supported in DI-drive.
For more details, check the `API docs <../api_doc/utils.html>`_.

Some of the default config setting is as following:

.. code:: python

    planner = dict(
        type='basic',
        min_distance=5.0,
        resolution=5.0,
        fps=10.
    )

- type
    Type of local planner
- min_distance
    Distance to find waypoint ahead vehicle.
- resolution
    Distance between waypoints when tracking route.

.. note::

    You can define your own local planner in the simulator. Simply add the planner class in ``PLANNER_DICT``
    in simulators.


Sensor configurations
---------------------------

Here we explain the 'obs' and 'aug' key in config

Sensor config is used to set up sensors and get image observation in simulation. There are several types of
sensors and images available to add into config list. You can add more than one sensor of same type.
They are tagged by provided 'name' key in config dict.

DI-drive simulator supports several types of sensor in Carla, including 'rgb' camera, 'depth' camera, 'segmentation' camera,
and 'lidar' sensors. We also add 'bev' image to paint a bird-eye view image for the hero actor. It contains
following information: road lines, lane lines, traffic lights, other vehicles, pedestrians, hero vehicles,
forward waypoints list. Each stored in one channel (traffic lights in three channels). The simulator can only
have one bird-eye view image to draw and store in sensor data. Here are the default configs for these sensors:

.. code:: python

    camera = dict(
        size=[384, 160],
        fov=90,
        position=[2.0, 0.0, 1.4],
        rotation=[0, 0, 0],
    )

    lidar = dict(
        channels=1,
        range=2000,
        points_per_second=1000,
        rotation_frequency=10,
        upper_fov=-3,
        lower_fov=-3,
        position=[0, 0.0, 1.4],
        rotation=[0, -90, 0],
        draw=False,
    )

    birdview = dict(
        size=[320, 320],
        pixels_per_meter=5,
        pixels_ahead_vehicle=100,
    )

To add these sensors in simulator, you can add elements into 'obs' config list and change their default value:

.. code:: python

    obs=[
        dict(
            name='front_rgb',
            type='rgb',
        ),
        dict(
            name='top_rgb',
            type='rgb',
            position=[-2.8, 0, 5.5],
            rotation=[-15, 0, 0],
        ),
        dict(
            name='birdview',
            type='bev',
        ),
    ]

Then you can have 'front_rgb', 'top_rgb', 'birdview' image added into sensor data.

Cameras may need to add augmentation. for example, there may be two cameras with different type and same position
to add random augmentation. So we need to store an augmentation setting in 'aug' key, which specifies the random
value range of the augmentation.

.. code:: python

    aug=dict(
        position_range=[0, 0, 0],
        rotation_range=[0, 0, 0],
    )

Running status
===================

DI-drive simulator can get and record some running status in each frame, including basic status, navigation,
information about simulation. They are automatically checked in each frame. Simulator provides interfaces to
get dictionary about these info, stored in each key.

Status
---------------

- speed (float): speed in km/h
- location (np.ndarray): 3D location vector
- forward_vector (np.ndarray): 2D forward vector
- acceleration (np.ndarray): 3D acceleration vector
- is_junction (bool): Whether is at junction in Carla map
- tl_state (int): Traffic light state

    * 0: Red
    * 1: Yellow
    * 2: Green
    * 3: Off
- tl_dis (float): Distance between vehicle and current road end

Navigation
------------

- agent_state (int): Agent navigation state

    * -1: Void
    * 1: Navigating
    * 2: Blocked by vehicle
    * 3: Blocked by walker
    * 4: Blocked by red light
- command (int): Road option during navigation

    * -1: Void
    * 1: Turn left
    * 2: Turn right
    * 3: Go straight at junction
    * 4: Follow lane
    * 5: Change lane to left
    * 6: Change lane to right
- node (np.ndarray): 2D location of current waypoint 
- node_forward (np.ndarray): 2D direction of current waypoint
- target (np.ndarray): 2D location of target waypoint(the next of node)
- target_forward (np.ndarray): 2D direction of target waypoint 
- waypoint_list (np.ndarray): Array containing waypoints on the route ahead. Each element is a 4D vector, consisting of the position and forward vector in x-y coordinates
- speed_limit (float): Speed limit of current position. It may come from the limit in current road and computed by planner

Information
---------------

- tick (int): Ticks for time in current episode
- timestamp (float): Real timestamp run in current episode
- total_lights (int): Total traffic lights met before
- total_lights_ran (int): Total red traffic light ran before
