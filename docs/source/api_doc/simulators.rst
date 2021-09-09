simulators
#################

.. toctree::
    :maxdepth: 2


BaseSimulator
===============
.. autoclass:: core.simulators.base_simulator.BaseSimulator
    :members:


CarlaSimulator
===============
.. autoclass:: core.simulators.CarlaSimulator
    :members:

.. warning::
    
    Make sure to use :func:`clean_up <core.simulators.carla_simulator.CarlaSimulator.clean_up>` before delete an simulator. 
    If not, all current actors(vehicles, pedestrians, sensors ect.) will
    remain in Carla client, and may cause error when next simulator link to that client!
        

CarlaScenarioSimulator
========================
.. autoclass:: core.simulators.CarlaScenarioSimulator
    :members:

