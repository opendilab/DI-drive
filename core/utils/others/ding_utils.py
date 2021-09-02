from ding.utils import deep_merge_dicts


def compile_config(cfg, env_manager, policy, learner, collector, buffer):
    cfg.env_manager = deep_merge_dicts(env_manager.default_config(), cfg.env_manager)
    cfg.policy = deep_merge_dicts(policy.default_config(), cfg.policy)
    cfg.policy.learn.learner = deep_merge_dicts(learner.default_config(), cfg.policy.learn.learner)
    cfg.policy.collect.collector = deep_merge_dicts(collector.default_config(), cfg.policy.collect.collector)
    cfg.policy.other.replay_buffer = deep_merge_dicts(buffer.default_config(), cfg.policy.other.replay_buffer)
    return cfg
