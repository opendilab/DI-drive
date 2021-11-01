Carla tutorial
################

.. toctree::
    :maxdepth: 2

Carla is currently the mainly used simualtor in DI-drive. This page shows some bacis concepts and
a quick user guide to run Carla simulator, for quick hands-on use of Carla in DI-drive

Basic Concept of Carla
=======================

Carla simulator runs in a client-server architecture. The sever and clients run individually. The server
contains map, weather, actors and all physical simulation. The clients can control the actors by
sending commands and ticking world in Carla server by its Python API. The server and clients
communicate through TCP link, consisting of a host and port provided by server, which can be specified when
establishing the server.

.. code:: bash

    ./CarlaUE4.sh --carla-world-port=N

Usually `N` is set as an even number, because the port `N+1` is simultanously occupied to connect the server. Carla 0.9.9.4
also uses a Traffic Manager port to control global behaviors of vehicles. It is specified at the
client side. TM client can not only be built by DI-drive via automatically finding free port in current system,
but also manually set by users.

.. code:: python

    carla_env = SimpleCarlaEnv(env_cfg, host, port, tm_port)

Carla simulation can run in **sync** mode or **async** mode. In **asynchronous** mode, the server runs
as fast as possible regardless of the client. In **synchronous** mode, the server and client runs in the
same fixed time step (common set as 0.1s), while the server will wait for client’s tick before updating
next step. For reasonable IL and RL trainning, DI-drive set Carla client and server to synchronous mode by default.

Carla allows multiple clients linking to one server. So make sure your Carla server only has one client
linked or only one client is sending command, in case of entangled or chaotic simulation.

When a client is destroyed, the actors and weather it create/change will NOT be released or reset.
So make sure to call the ``clean_up`` method in simulator to manually destroy all actors and reset weather
changed by the client.


Quickly Create Carla server via Docker
========================================

It is recommended to run Carla server in `docker <https://www.docker.com>`_ which makes the installation
much easier and can run multiple Carla servers. DI-drive follows guidance in Carla doc which uses
`docker CE <https://docs.docker.com/engine/install/ubuntu>`_ and
`nvidia docker2 <https://github.com/NVIDIA/nvidia-docker>`_.

Then, just pull the Carla image and run with port.

.. code:: bash

    docker pull carlasim/carla:0.9.9.4
    # by default
    docker run -p 2000-2002:9000-9002 --runtime=nvidia --gpus <gpu_id> carlasim/carla:0.9.9.4
    # with parameters
    docker run -p 2000-2002:9000-9002 --runtime=nvidia --gpus <gpu_id> carlasim/carla:0.9.9.4 /bin/bash CarlaUE4.sh <list of paremeters>

We also provide an easily used multi-carla docker image that can start an amount of Carla servers.
You can pull the image from `dockerhub <https://hub.docker.com>`_ and start with your own settings.
For example, the following command will start a container with 8 Carla server whose ports are set
from 9000 to 9014

.. code:: bash

    docker pull opendilab/multi-carla:0.9.9
    docker run -p 9000-9016:9000-9016 --runtime=nvidia opendilab/multi-carla:0.9.9 /bin/bash run_carla.sh -n 8 -p 9000

Generally, the option ``-n NUMS`` sets the number of Carla server and the option ``-p PORT`` sets
the start port of servers. The other servers' port are set to the following even number.

.. note::

    Please pay attention to the Carla port within and out of the container. The internal ones are set
    by the parameters in carla scripts, the external ones are set when creating the container.


Carla server settings
===========================

We provide config setting to quickly set host and port for several Carla servers. The ``server`` key in config
file is used to set Carla servers. For example:

.. code::

    config = dict(
        server=[
            dict(carla_host='192.168.1.1', carla_ports=[5000, 5010, 2]),
            dict(carla_host='localhost', carla_ports=[9000, 9010, 2]),
        ],
        ...
    )

You can use Carla servers in several IP hosts, each is stored as an element in ``server`` list. Each IP has a
host and a list of ports. The interval between ports is 2, because Carla needs two ports to communicate. By
defalut the `N+1` port is occupied. The saving form of ports in config is similar to the Python ``range`` method.
:code:`carla_ports=[9000, 9010, 2]` means there are 5 Carla servers whose ports are '9000, 9002, 9004, 9006
, 9008'.
