.. _header-n32:

Conditional Imitation Learning
===================================

.. toctree::
    :maxdepth: 2

`Conditional Imitation Learning(CIL) <https://arxiv.org/abs/1710.02410>`_ is a
widely used imitation learning method for autoploit. The CIL takes front
RGB camera image and vehicle speed as inputs, then predict vehicle control
singals including steer, throttle and brake under certain navigation
command.

.. figure:: ../../figs/image-cils_1.png
   :alt: image-cilrs_1
   :align: center

The CIL model use convolutional neural network to capture visual
information and concat it with measurements information (like
vehicle speed), then use the navigation command to select a suitable
branch to give prediction. More details about CIL can be found in the
paper.

.. figure:: ../../figs/image-cils_2.png
   :alt: image-cilrs_2
   :align: center

DI-drive supports full pipeline of CIL, including datasets collecting, model
training, and benchmark evaluation. Quick start of CIL has been
introduced in the tutorial. All code in this page can be found in ``./demo/coil_demo``

.. _header-n66:

Data collection
---------------

The configuration of dataset collection has been set up in
``coil_data_collect.py.`` You can custom it by
modifying the configuration.

.. code:: python

    config = dict(
        env=dict(
            env_num=5,
            simulator=dict(
                disable_two_wheels=True,
                waypoint_num=32,
                planner=dict(
                    type='behavior',
                    resolution=1,
                ),
                obs=(
                    dict(
                        name='rgb',
                        type='rgb',
                        size=[800, 600],
                        position=[2.0, 0.0, 1.4],
                        rotation=[-15, 0, 0],
                    ),
                ),

            ),
            col_is_failure=True,
            stuck_is_failure=True,
            manager=dict(
                auto_reset=False,
                shared_memory=False,
                context='spawn',
                max_retry=1,
            ),
            wrapper=dict(),
        ),
        server=[
            dict(carla_host='localhost', carla_ports=[9000, 9010, 2]),
        ],
        policy=dict(
            target_speed=25,
            noise=True,
            collect=dict(
                n_episode=5,
                dir_path='./datasets_train/cils_datasets_train',
                collector=dict(
                    suite='FullTown01-v1',
                ),
            )
        ),
    )

.. _header-n75:

Model training
--------------

The training of CIL is a standard imitation learning process. It takes
RGB image and measurement as input then mimic the export label. You can custom
the training by changing model architecture, learning rate, steps and so
on in ``coil_train.py``

.. code:: python

    train_config = dict(
        NUMBER_OF_LOADING_WORKERS=12,
        SENSORS=dict(rgb=[3, 88, 200]),
        TARGETS=['steer', 'throttle', 'brake'],
        INPUTS=['speed_module'],
        BATCH_SIZE=120,
        COMMON=dict(
            folder='sample', exp='coil_icra', dataset_path='./datasets_train'
        ),
        GPU='0',
        SAVE_SCHEDULE=[0, 100, 1000001],
        NUMBER_ITERATIONS=200000,
        SPEED_FACTOR=25.0,
        TRAIN_DATASET_NAME='cils_datasets_train',
        AUGMENTATION=None,
        NUMBER_OF_HOURS=50,
        MODEL_TYPE='coil-icra',
        MODEL_CONFIGURATION=dict(
            perception=dict(res=dict(name='resnet34', num_classes=512)),
            measurements=dict(fc=dict(neurons=[128, 128], dropouts=[0.0, 0.0])),
            join=dict(fc=dict(neurons=[512], dropouts=[0.0])),
            speed_branch=dict(fc=dict(neurons=[256, 256], dropouts=[0.0, 0.5])),
            branches=dict(number_of_branches=4, fc=dict(neurons=[256, 256], dropouts=[0.0, 0.5]))
        ),
        PRE_TRAINED=False,
        LEARNING_RATE_DECAY_INTERVAL=75000,
        LEARNING_RATE_DECAY_LEVEL=0.1,
        LEARNING_RATE_THRESHOLD=5000,
        LEARNING_RATE=0.0002,
        BRANCH_LOSS_WEIGHT=[0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.05],
        VARIABLE_WEIGHT=dict(Steer=0.5, Gas=0.45, Brake=0.05),
        LOSS_FUNCTION='L1',
        IMAGE_CUT=[115, 500],
        PRELOAD_MODEL_ALIAS=None,
        PRELOAD_MODEL_BATCH=None,
        PRELOAD_MODEL_CHECKPOINT=None,
        REMOVE=None,
        NUMBER_IMAGES_SEQUENCE=1,
        NUMBER_FRAMES_FUSION=1,
    )

.. _header-n79:

Model evaluation
----------------

DI-drive supports model evaluation with Carla benchmark. To custom the
evaluation process, you can modify the configuration in ``coil_eval.py``.

.. code:: python

    autoeval_config = dict(
        env=dict(
            env_num=5,
            simulator=dict(
                verbose=False,
                obs=(
                    dict(
                        name='rgb',
                        type='rgb',
                        size=[800, 600],
                        position=[2.0, 0.0, 1.4],
                        rotation=[-15, 0, 0],
                    ),
                ),
                planner=dict(type='behavior', ),
            ),
            manager=dict(
                shared_memory=False,
                auto_reset=False,
                context='spawn',
                max_retry=1,
            ),
        ),
        server=[
            dict(carla_host='localhost', carla_ports=[9000, 9010, 2])
        ],
        policy=dict(
            target_speed=40,
            eval=dict(
                evaluator=dict(
                    suite='FullTown02-v1',
                    episodes_per_suite=5,
                    save_files=True,
                ),
            ),
        ),
    )

    policy_config = dict(
        model=dict(
            ckpt_path='./coil_icra/checkpoints/100.pth'
        ),
        SENSORS=dict(rgb=[3, 88, 200]),
        TARGETS=['steer', 'throttle', 'brake'],
        INPUTS=['speed_module'],
        SPEED_FACTOR=25.0,
        MODEL_TYPE='coil-icra',
        MODEL_CONFIGURATION=dict(
            perception=dict(res=dict(name='resnet34', num_classes=512)),
            measurements=dict(fc=dict(neurons=[128, 128], dropouts=[0.0, 0.0])),
            join=dict(fc=dict(neurons=[512], dropouts=[0.0])),
            speed_branch=dict(fc=dict(neurons=[256, 256], dropouts=[0.0, 0.5])),
            branches=dict(number_of_branches=4, fc=dict(neurons=[256, 256], dropouts=[0.0, 0.5]))
        ),
        PRE_TRAINED=False,
        LEARNING_RATE_DECAY_LEVEL=0.1,
        IMAGE_CUT=[115, 500],
        NUMBER_FRAMES_FUSION=1,
    )

.. code:: 

   @inproceedings{2018End,
     title={End-to-End Driving Via Conditional Imitation Learning},
     author={ Codevilla, Felipe  and  Miiller, Matthias  and  Lopez, Antonio  and  Koltun, Vladlen  and  Dosovitskiy, Alexey },
     booktitle={2018 IEEE International Conference on Robotics and Automation (ICRA)},
     year={2018},
   }
