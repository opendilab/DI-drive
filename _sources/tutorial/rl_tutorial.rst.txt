Simple Reinforcement Learning
##############################

.. toctree::
    :maxdepth: 2

Prerequisites
----------------

Ubuntu 16.04 system +Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz + 32G
memory + GPU1060


DI-drive RL training using DI-engine
--------------------------------------

DI-drive + DI-engine make RL for Autonomous Driving very easy. We build a simple RL demo that can run
varies RL algorithm with a simple environment setting. It takes a small Bird-eye View image as NN
input, together with current speed, and directly output control signals. All the code can be found in
``demo/simple_rl``.

Here we show how to run the DQN demo. It follows the standard deployment of a DI-engine RL entry.
Other RL demo is written in same way.

.. code:: bash

    cd demo/simple_rl
    python dqn_main.py

The config part defines the env and policy settings. Notes that you need to change the Carla server
host and port, and modify the environment nums according to yours. By default it uses 8 Carla server on
`localhost` with port from 9000 to 9016.

.. code:: python

    train_config = dict(
        env=dict(
            ...
        ),
        env_num=8,
        server=[
            dict(carla_host='localhost', carla_ports=[9000, 9016, 2]),
        ],
        policy=dict(
            ...
        ),
    )

For more details about how to tune parameters in DQN, you can see their doc. Usually
you may concern about the replay buffer size and sample num per collection.

When you see the information in terminal that contains the content in
the following picture, it means that you are beginning to train the
model.

.. figure:: ../../figs/rl_tutorial_log.png
    :alt: rl_tutorial_log
    :align: center
    :width: 800px

In the process of training, you can use the tensorboard as a monitor,
the default log path is in your working directory.

.. code:: bash

    tensorboard --logdir='./log'

After running for about 24 hours, you will get:

.. figure:: ../../figs/rl_tutorial_tb.png
    :alt: rl_tutorial_tb
    :align: center
    :width: 800px
      
Evaluate the trained model
----------------------------

After training, you can evaluate the trained model with a vsiualization screen. Simply run the following code.

.. code:: bash

    cd demo/simple_rl
    python dqn_eval.py

You may need to change Carla server, change the suite you want to evaluate, and switch on/off visualization or save a replay gif/video.
You can add your pre-trained weights in policy's config.

.. code:: python

    eval_config = dict(
        env=dict(
            ...
            visualize=dict(
                type='birdview',
                outputs=['show'], # or 'gif', 'video'
                save_dir='',
                frame_skip=3, # avoid to be too large
            ),
        ),
        server=[dict(
            carla_host='localhost',
            carla_ports=[9000, 9002, 2]
        )],
        policy=dict(
            cuda=True,
            ckpt_path='path/to/your/model',
        ),
        ...
    )

The default DQN policy can have nice probability to complete navigation in `FullTown02-v2`, with traffic lights
ignored.
