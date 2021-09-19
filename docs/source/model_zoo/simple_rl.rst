BeV Speed End-to-end Reinforcement Learning
##############################################

.. toctree::
    :maxdepth: 2

This is a simple Reinforcement Learning demo to show the basic usage of DI-drive environments
and DI-engine RL policies.


Inputs, Env and NN models
=====================================

We use a Bird-eye View image with size of 32x32x5 and speed scalar as observations. The BeV image
is encoded by a conv net to get a 256 size embedding, and concat with the speed value which is
repeated 256 times.

The encoder output is then send into different heads depends on the required outputs of RL policies.
Currently we have DQN, DDPG, TD3, SAC and PPO demos. We provide training and evaluation entry for
all of them.

You can refer to the defination of encoder in ``core/models/bev_speed_model.py`` and RL models in
``demo/simple_rl/model.py`` for their details. If you want to build your own RL experiments, you can
define NN models similarly.

The environment instance ``SimpleCarlaEnv`` is well defined with specified inputs and outputs. 
The standard usage to customize the env interfaces is to add ``EnvWrapper`` and change the input, output
of Env. For example:

.. code:: python

    class DiscreteEnvWrapper(gym.Wrapper):

        self._acc_list = [(0, 1), (0.25, 0), (0.75, 0),]
        self._steer_list = [-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8]

        def reset(self, *args, **kwargs) -> Any:
            obs = self.env.reset(*args, **kwargs)
            obs_out = {
                'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
                'speed': (obs['speed'] / 25).astype(np.float32),
            }
            return obs_out

        def step(self, id):
            if isinstance(id, torch.Tensor):
                id = id.item()
            id = np.squeeze(id)
            assert id < len(self._acc_list) * len(self._steer_list), (id, len(self._acc_list) * len(self._steer_list))
            mod_value = len(self._acc_list)
            acc = self._acc_list[id % mod_value]
            steer = self._steer_list[id // mod_value]
            action = {
                'steer': steer,
                'throttle': acc[0],
                'brake': acc[1],
            }
            timestep = self.env.step(action)
            obs = timestep.obs
            obs_out = {
                'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
                'speed': (obs['speed'] / 25).astype(np.float32),
            }
            timestep = timestep._replace(obs=obs_out)
            return timestep

This will map the descrete action space to continuous control signal in Carla env and delete
traffic signal channels in BeV image. Other wrappers work in the same way.


Training loop
===================

The entry files of all the RL methods are written in standard way using DI-engine to run RL experiments.
The sub-process env manager in Di-engine is used to run multi-env in parallel.
Off-policy methods use collector, learner, replay buffer in DI-engine and evaluator in DI-drive. On-policy
method does not use replay buffer.

Off-policy training loop:

.. code:: python

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, rate = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        update_per_collect = len(new_data) // cfg.policy.learn.batch_size * 4
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Training
        for i in range(update_per_collect):
            train_data = replay_buffer.sample(cfg.policy.learn.batch_size, learner.train_iter)
            if train_data is not None:
                train_data = copy.deepcopy(train_data)
                unpack_birdview(train_data)
                learner.train(train_data, collector.envstep)
            replay_buffer.update(learner.priority_info)


Env Wrappers
=================

The policy is trained in single-lane maps in Carla. Town01 for training and Town02 for evaluation.
The traffic lights are ignored and also the traffic light channels in BeV images is deleted.

The environment instance ``SimpleCarlaEnv`` is well defined with specified inputs and outputs. 
The standard usage to customize the env interfaces is to add ``EnvWrapper`` and change the input, output
of Env. For example:

.. code:: python

    DEFAULT_ACC_LIST = [(0, 1), (0.25, 0), (0.75, 0),]
    DEFAULT_STEER_LIST = [-0.8,-0.5,-0.2,0,0.2,0.5,0.8,]

    class DiscreteEnvWrapper(gym.Wrapper):

        self._acc_list = [(0, 1), (0.25, 0), (0.75, 0),]
        self._steer_list = [-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8]

        def reset(self, *args, **kwargs) -> Any:
            obs = self.env.reset(*args, **kwargs)
            obs_out = {
                'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
                'speed': (obs['speed'] / 25).astype(np.float32),
            }
            return obs_out

        def step(self, id):
            if isinstance(id, torch.Tensor):
                id = id.item()
            id = np.squeeze(id)
            assert id < len(self._acc_list) * len(self._steer_list), (id, len(self._acc_list) * len(self._steer_list))
            mod_value = len(self._acc_list)
            acc = self._acc_list[id % mod_value]
            steer = self._steer_list[id // mod_value]
            action = {
                'steer': steer,
                'throttle': acc[0],
                'brake': acc[1],
            }
            timestep = self.env.step(action)
            obs = timestep.obs
            obs_out = {
                'birdview': obs['birdview'][..., [0, 1, 5, 6, 8]],
                'speed': (obs['speed'] / 25).astype(np.float32),
            }
            timestep = timestep._replace(obs=obs_out)
            return timestep

This will map the descrete action space to continuous control signal in Carla env and delete
traffic signal channels in BeV image. Other wrappers work in the same way.

