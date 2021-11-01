.. DI-drive documentation master file, created by
   sphinx-quickstart on Mon Jan 25 13:49:15 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DI-drive Documentation
=========================

.. toctree::
    :maxdepth: 2

.. figure:: ../figs/DI-drive.png
   :alt: DI-drive
   :width: 300px

Decision Intelligence Platform for Autonomous Driving simulation.

Last updated on 2021.10.28

-----

**DI-drive** is an open-sourece application platform under **OpenDILab**.
**DI-drive** applies Deep Learning method to decision, planning and controlling task in Autonomous Driving simulation,
with **high ease of use** and **low difficulty to hands-on**.
**DI-drive** designes **Casezoo** to make simulation environment closer to real driving.
**DI-drive** develops Deep Learning driving policy with `Pytorch <http://pytorch.org>`_ and `DI-engine <https://github.com/opendilab/DI-engine>`_
and mainly uses `Carla <http://carla.org>`_ simulator.

Main features for **DI-drive**
----------------------------------

- **AD Simulation**
   **DI-drive** provides unified and easily used interfaces to support all kinds of lightweight and complex driving simulators.
   The user only needs to customize the input and output of the driving policy and simply call these interfaces to complete the simualtion.

- **Simplified training process**
   **DI-drive** provides a variety of modules and tools to ease the training and testing of driving policies, including data collection,
   training and evaluation for Imitation Learning, standard ``gym.Env`` instance for Reinforcement Learning, and simple usage of **DI-engine**.
   This can greatly reduce the difficulty of deploying driving policy for beginners.

- **Modular Design**
   Autonomous driving tasks are decomposed into *Policy* and *Environment*. Users can customize and modify simulation and training settings by changing the configuration files,
   without diving deep into complex models and internal details of the simulator. **DI-drive** defines policy in a flexible, polymorphic and efficient way.
   It can adapt to all existing academic literatures and support complex training tasks across methods, models, datasets, and even across simulators.

- **CaseZoo sets**
   **DI-drive** designs a **Casezoo** simulation set, by integrating existing Autonomous Driving evaluation indicators, scenarios and tools in both academia and industry.
   **Casezoo** combines data collected by real vehicles and Shanghai Lingang road license test Scenarios.
   **Casezoo** can realize scenario-based Autonomous Driving test, makes the simulation closer to real driving.
   **Di-drive** supports to test as well as train Renfircement Learning policy with **Casezoo** environment.

------

This is home page contains description of tutorials, features and API documentation for **DI-drive**. Feel free 
to check any part you need. It is recommended to first install **DI-drive** and have a quick try following 
provided quick start guidence.

Table of Contents
--------------------


.. toctree::
   :maxdepth: 2

   installation/index

.. toctree::
   :maxdepth: 3

   tutorial/index

.. toctree::
   :maxdepth: 2

   features/index


.. toctree::
   :maxdepth: 2

   model_zoo/index


.. toctree::
   :maxdepth: 1

   api_doc/index


.. toctree::
   :maxdepth: 1

   faq/index