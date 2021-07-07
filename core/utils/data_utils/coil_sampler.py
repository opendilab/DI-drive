import random

from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):
    r"""Samples elements randomly from a given list

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, keys, cfg, executed_iterations):
        self.iterations_to_execute = ((cfg.NUMBER_ITERATIONS) * cfg.BATCH_SIZE) -\
                                     (executed_iterations)

        self.keys = keys

    def __iter__(self):

        return iter([random.choice(self.keys) for _ in range(self.iterations_to_execute)])

    def __len__(self):
        return self.iterations_to_execute
