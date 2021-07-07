import torch

from .coil_sampler import RandomSampler


def select_balancing_strategy(dataset, cfg, iteration):
    number_of_workers = cfg.NUMBER_OF_LOADING_WORKERS
    keys = range(0, len(dataset) - cfg.NUMBER_IMAGES_SEQUENCE)
    sampler = RandomSampler(keys, cfg, iteration * cfg.BATCH_SIZE)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE, sampler=sampler, num_workers=number_of_workers, pin_memory=True
    )
    return data_loader
