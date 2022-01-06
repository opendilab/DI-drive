from easydict import EasyDict

ppo_config = dict(
    exp_name='ppo21_bev32_lr1e4_bs128_ns3000_update5_train_ft',
    env=dict(
        collector_env_num=7,
        evaluator_env_num=1,
        simulator=dict(
            town='Town01',
            disable_two_wheels=True,
            verbose=False,
            waypoint_num=32,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='birdview',
                    type='bev',
                    size=[32, 32],
                    pixels_per_meter=1,
                    pixels_ahead_vehicle=14,
                ),
            ),
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        ignore_light=True,
        replay_path='./ppo_video',
        visualize=dict(
            type='birdview',
        ),
        manager=dict(
            collect=dict(
                auto_reset=True,
                shared_memory=False,
                context='spawn',
                max_retry=2,
                retry_type='renew',
                step_timeout=120,
                reset_timeout=120,
            ),
            eval=dict()
        ),
        wrapper=dict(
            # Collect and eval suites for training
            collect=dict(suite='train_ft', ),
            eval=dict(suite='FullTown02-v1', ),
        ),
    ),
    server=[
        dict(carla_host='localhost', carla_ports=[9000, 9016, 2]),
    ],
    policy=dict(
        cuda=True,
        nstep_return=False,
        on_policy=True,
        model=dict(),
        learn=dict(
            epoch_per_collect=5,
            batch_size=128,
            learning_rate=0.0001,
            weight_decay=0.0001,
            value_weight=0.5,
            adv_norm=False,
            entropy_weight=0.01,
            clip_ratio=0.2,
            target_update_freq=100,
            learner=dict(
                hook=dict(
                    log_show_after_iter=1000,
                    load_ckpt_before_run='',
                ),
            ),
        ),
        collect=dict(
            collector=dict(
                collect_print_freq=1000,
                deepcopy_obs=True,
                transform_obs=True,
            ),
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=5000,
                n_episode=5,
                stop_rate=0.7,
                transform_obs=True,
            ),
        ),
    ),
)

default_train_config = EasyDict(ppo_config)