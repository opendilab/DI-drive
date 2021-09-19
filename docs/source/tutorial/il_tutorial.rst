.. _header-n2:

Simple Imitation Learning
==========================

Here we will show you how to use **DI-drive** for imitation learning research
of autopilot. **DI-drive** supports 
`Conditional Imitation Learning(CIL) <http://vladlen.info/papers/conditional-imitation.pdf>`__
for Imitation Learning research.

In this tutorial, we will show how to use **DI-drive** to train a CIL model.
CIL takes front RGB camera and vehicle speed as input, then output the prediction of vehicle control singals including
steer, throttle and brake under certain navigation command.


IL of autopilot normally includes three parts:

-  Datasets collection

-  Model training

-  Model evaluation

For now, we will show you step by step

.. _header-n14:

Datasets collection
-------------------

To collect the datasets, the first step is to start a Carla server

.. code:: shell

   ./CarlaUE4.sh -fps=10 -benchmark -world-port=[PORT NUM]

To customize your training datasets, please refer to the configuration
``coil_data_collect.py``. For this tutorial, you
should change the ``save_dir`` path and Carla server host\port.

.. code:: python

    saver=dict(save_dir='<path_to_your_dataset>/cils_datasets_train'),
    server=[
            dict(carla_host='localhost', carla_ports=[9000, 9008, 2]),
        ],

Then you can use the python scrips to collect the data.

.. code:: shell

   cd demo/coil_demo
   python coil_data_collect.py 

This collecting process usually cost ~7 hours and ~12 G storage with
single Carla thread. You can speed up this process by starting multi-carla envs.


If you finished this step, there should be a ``.npy`` file under
``_preloads``.

Once you finish the data collection, the next step is to train an CIL model.

.. _header-n26:

Model training
--------------

The training part of IL enables a model to mimic the
output of an expert policy with the same input. To apply the model training
with **DI-drive** with default training configuration, just run

.. code:: 

   python coil_train.py

Note that you should modify the dataset path as you have created
above.

.. code:: python

                   COMMON=dict(folder='sample', exp='coil_icra',
                                dataset_path='<path_to_your_dataset>/datasets_train')

You will see loss printed on the screen like

.. figure:: ../../figs/image-il_1.png
   :alt: image-il_1
   :align: center
   :width: 300px

and the training checkpoints will be saved under ``_logs`` folder. The
training process will last for 7~8 hours (i7-9900k and GeForce RTX 2080Ti), up to your machine hardware.

if you want to customize your training configuration such as the model architecture and
learning rates, please modify the config in ``coil_train.py``

.. _header-n35:

Model evaluation
----------------

To evaluate your trained model with **DI-drive**, just run

.. code:: python

   python coil_eval.py 

Note that you have change the checkpoint path and Carla server host/port in ``coil_eval.py``

.. code:: python

    ckpt_path='<path_to_your_checkpoint>'
    server=[dict(carla_host='localhost', carla_ports=[9000, 9010, 2])],

You can visual the real-time evaluation process like

.. figure:: ../../figs/image-il_2.png
   :alt: image-il_2
   :align: center
   :width: 500px

When the evaluation is finished, you will get a performance table.

You can customize the evaluation configuration by modifying the config in 
``coil_eval.py``

Congratulations! You have finished the imitation learning tutorial.
