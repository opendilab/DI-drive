Latent Reinforcement Learning
#################################

.. toctree::
    :maxdepth: 2

Latent Reinforcement Learning is a category of policy settings which generate
latent representations that can completely characterize the features related to
the driving task in the input observation. Techically, it uses a Variational
Autoencoder (VAE) to get latent embeddings, and takes it as observation to
train RL policies.

Our implementation refers to `Model-free RL for AD <https://arxiv.org/abs/1904.09503>`_
with **DI-drive** and **DI-engine** and transfers it to more general cases. All
entries can be found in ``demo/latent_rl``. In the
following documents, it will be explained detailly.


Training auto-encoder
======================

The first step is to train a VAE to encode input image
into a imbedding feature. We deploy the model and training following
open source VAE `vae <https://github.com/AntixK/PyTorch-VAE>`__. 
..You can also use pretrained model vae.ckpt.
We follow the standard setting. The input of the auto-encoder module
is birdview image, whose shape is 192 * 192 * 7. We use 5
convolutional layers activated with leakyReLU as the encoder, which
encode the input birdview image to a embedding with 128 channels.
The decoder shares the same convolution kernel size and activation
layer, but uses transposed convolutional layer. Using the encoder,
we can convert the input birdview image to 128-channel embedding,
which helps the training of RL agent.

The training procedure includes collecting birdview data and training the
auto-encoder.

.. code:: bash

    cd demo/latent_rl
    # Collect data
    python collect_data.py
    # Train VAE
    python train_vae.py

You can custom the configurations in the files above.

Trainging RL agent
========================

Following Model-free RL for AD , we use DDQN as the
discrete RL algorithm and TD3 as the continous RL algorithm.
For DDQN, we discrete the steering and throttle to 10 class, which
means the whole action space is 100. The steering and throttle is
mapping to [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8] and
[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] respectively.
although the action space is discrete, the vehicle can run smoothly when
turning.
For TD3, we use continuous action space. The policy network's ouput is
the mean and log scale standard deviation of a Gaussion distribution.
At each state of collecting data, we sample an action following the
generated Gaussion distribution. For TD3, we found that the steering
changes drasticly. The steering is -1 or 0.8 and the throttle is
always 0.9. However, the car can drive well. This is a point to be
improved.

The training, evaluation and testing of the method mentioned above is
provided. We use the standard policy settings of **DI-engine**. You can
see their doc to get how to modify the training settings. You may need
to change the Carla server setting in the entry files.

DDQN:

.. code:: bash

    # Train DDQN agent
    python latent_dqn_train.py
    # Benchmark evaluation
    python latent_dqn_eval.py
    # Test and visualize
    python latent_dqn_test.py

Pending
