Datasets
###########

.. toctree::
    :maxdepth: 2

Benchmark Datasets
===================

DI-drive defines a unified benchmark dataset format that makes data collection and loading procedure easily for users.
It is suggested to save datasets with :class:`BenchmarkDatasetSaver <core.data.dataset_saver.BenchmarkDatasetSaver>`.
It can automatically create folders and save all sensor data and measurements into datasets as desired form.

General structure
----------------------

A dataset directory should look like the structure below. The following sections
will explain each one of the components described.

.. code::

    <dataset_name>
    │   dataset_metadata.json
    │
    └───episode_00000
    │   │   episode_metadata.json
    │   │   <Camera1_name>_00000.png
    │   │   ...
    │   │   <Camera2_name>_00000.png
    │   │   ...
    │   │   <Lidar_name>_00000.png
    │   │   ...
    │   │   measurements_00000.lmdb
    │   │   ...
    └───episode_00001
        │   ...
        │
        ...

A dataset contains dataset metadata, episode metadata, sensor data and measurements.

Dataset metadata
-----------------

Each dataset contains a metadata file with information provided by user.
It may have the following contents:

* Number of episodes
* Collected suite
* Obs image types and names

Episode metadata
------------------

Each episode is stored in a folder.
For each collected episode we generate a json file containing
its general aspects that are:

* Town map name.
* Start and end waypoint indexes
* Number of Pedestrians: the total number of spawned pedestrians.
* Number of Vehicles: the total number of spawned vehicles.
* Spawned seed for pedestrians and vehicles: the random seed used for the CARLA object spawning process.
* Weather: the weather of the episode.

Each episode lasts from 1-5 minutes
partitioned in simulation steps of 100 ms.
For each step, we store data divided
into two different categories, sensor data
stored as PNG images, and measurement data stored as json files.

Sensor data
------------

All images collected are stored as png files. The name consists of its tag in observation configurations
and the frame number.

Measurements
--------------

Measurements represent all the float data collected for each simulation
step. Each measurement arranges its content in a fixed order,
and stores them in a .lmdb file. The content is shown follow:

* tick (int)
* timestamp (float)
* forward_vector (2D)
* acceleration (3D)
* location (3D)
* speed (float)
* command (int)
* steer (float)
* throttle (float)
* brake (float)
* real_steer (float)
* real_throttle (float)
* real_brake (float)
* tl_state (int)
* tl_dis (float)

Their meaning is the same as the observation returned in :class:`SimpleCarlaEnv <core.envs.simple_carla_env.SimpleCarlaEnv>`

Others
-----------

It is allowed to add user customized data into datasets. The data can be post-processed and stored in an 'other' key, 
with no effect to measurements. Users can organize their necessary information into datasets freely.
