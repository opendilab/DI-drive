from easydict import EasyDict

sac_config = dict(
    exp_name='sac2_bev32_buf2e5_lr1e4_bs128_ns3000_update4_train_ft',
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
            )
        ),
        col_is_failure=True,
        stuck_is_failure=True,
        ignore_light=True,
        replay_path='./sac_video',
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
        model=dict(
            action_shape=2,
            twin_critic=True
        ),
        learn=dict(
            batch_size=128,
            learning_rate_q=1e-4,
            learning_rate_policy=1e-4,
            learning_rate_value=1e-4,
            learning_rate_alpha=1e-4,
            weight_decay=0.0001,
            learner=dict(
                hook=dict(
                    log_show_after_iter=1000,
                    load_ckpt_before_run='',
                ),
            ),
        ),
        collect=dict(
            n_sample=3000,
            noise_sigma=0.1,
            collector=dict(
                collect_print_freq=1000,
                deepcopy_obs=True,
                transform_obs=True,
            ),
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=5000,
                n_episode=5,
                stop_rate=0.7,
                transform_obs=True,
            ),
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=400000,
                replay_buffer_start_size=10000,
                max_use=16,
                monitor=dict(
                    sampled_data_attr=dict(
                        print_freq=100,  # times
                    ),
                    periodic_thruput=dict(
                        seconds=120,
                    ),
                ),
            ),
        ),
    ),
)

default_train_config = EasyDict(sac_config)