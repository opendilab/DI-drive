Learning by Cheating
==========================

`Learning by Cheating(LBC) <https://arxiv.org/abs/1912.12294>`_
is an Imitation Learning method which driving in Carla using waypoint prediction and
two-stage training -- cheating model and target model. The cheating model takes
ground-truth Bird-eye View image as input to teach the RGB input target policy.
The prediction of the model is a waypoint list then the waypoint list is processed by
a PID controller to get the final control singals including steer,
throttle and brake.

.. figure:: ../../figs/image-lbc_1.png
   :alt: image-lbc_1
   :align: center

Datasets collection
-------------------

Pending

Model training 
--------------

Pending

Model evaluation
----------------

DI-drive provides benchmark evaluation for both cheating model and target model,
together with pretrain weights. You may need to change the Carla server settings
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
