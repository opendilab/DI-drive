Learning by Cheating
==========================

`Learning by Cheating(LBC) <https://arxiv.org/abs/1912.12294>`_
is an Imitation Learning method which driving in Carla using waypoint prediction and
two-stage training -- cheating model and target model. The cheating model takes
ground-truth Bird-eye View(BeV) image as input to teach the RGB input target policy.
The prediction of the model is a waypoint list then the waypoint list is processed by
a PID controller to get the final control signals including steer,
throttle and brake.

.. figure:: ../../figs/image-lbc_1.png
   :alt: image-lbc_1
   :align: center

   LBC structure

**DI-drive** provide complete pipeline of training an LBC privileged model and target
model, including data collection, birdview model training, image model training, mimicing
cheating model and evaluation. All entries can be found in ``demo/lbc``


Datasets collection
-------------------

**DI-drive** provides benchmark data collection for LBC. The dataset is formatted to save
rgb data as `.png` files and BeV data into `.lmdb` files. The configuration of dataset
collection is set up in ``lbc_collect_data.py``. You may need to change the Carla server,
dataset path, collection suites, planner configs and noise args etc.

.. code:: python

    config = dict(
        env=dict(
            env_num=5,
            simulator=dict(
                planner=dict(
                    ...
                ),
                ...
            ),
        ),
        server=[
            dict(carla_host='localhost', carla_ports=[5000, 5010, 2]),
        ],
        policy=dict(
            target_speed=25,
            noise=True,
            noise_kwargs=dict(
                ...
            ),
            collect=dict(
                dir_path='./datasets_train/lbc_datasets_train',
                n_episode=100,
                collector=dict(
                    suite='FullTown01-v1',
                    nocrash=True,
                ),
            ),
            ...
        ),
    )

You may need to change the Carla server numbers and ports, and dataset path to yours.

.. code:: bash

    python collect_data.py

Model training 
--------------

The training of LBC model contains 3 stages: training privileged model, offline
training target model, on-line fine-tuning target model.

Training privileged model
******************************

The training of privileged model is a supervise learning procedure. You can check
the training code in ``lbc_birdview_train.py``. By default it will save checkpoint
and tensorboard logs in ``./log``. You can check the training progress and effects.

.. code::

    python lbc_birdview_train.py


Offline training target model
********************************

The off-line training of target model consists of 2 procedures, first warm-up training
with dataset label, then mimicing with privileged model label. Both training codes are
provided in ``lbc_img_train_phase0.py`` and ``lbc_img_train_phase1.py``.

.. code::

    python lbc_img_train_phase0.py
    python lbc_img_train_phase1.py

You may need to change the dataset path, checkpoint path etc.


Online training target model
*******************************

Pending


Model evaluation
----------------

DI-drive provides benchmark evaluation for both cheating model and target model,
together with pre-trained weights. You may need to change the Carla server settings
to yours.

Pretrain weights: 
`bev <http://opendilab.org/download/DI-drive/lbc/birdview/model-256.th>`_, 
`image <http://opendilab.org/download/DI-drive/lbc/rgb/model-20.th>`_

.. code:: shell

    python lbc_bev_eval.py
    python lbc_image_eval.py

Then you will get the performance table.

Model testing
-----------------

DI-drive provides a simple entry for testing LBC models in a benchmark or Casezoo
environment and visualize the running. You may need to change the Carla server settings
to yours.

.. code:: shell

    python lbc_bev_test.py
    python lbc_image_test.py

.. code:: 

   @inproceedings{chen2019lbc,
     author    = {Dian Chen and Brady Zhou and Vladlen Koltun and Philipp Kr\"ahenb\"uhl},
     title     = {Learning by Cheating},
     booktitle = {Conference on Robot Learning (CoRL)},
     year      = {2019},
   }
