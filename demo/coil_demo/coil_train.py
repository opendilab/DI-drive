import os
import time

import torch
import torch.optim as optim
from easydict import EasyDict

from core.data.coil_dataset import CoILDataset
from core.models.coil_model import CoILModule
from core.utils.data_utils.splitter import select_balancing_strategy
from core.utils.learner_utils.loss_utils import Loss
from core.utils.learner_utils.optim_utils import adjust_learning_rate_auto
from core.utils.others.checkpoint_helper import is_ready_to_save, get_latest_saved_checkpoint
from core.utils.others.general_helper import create_log_folder, create_exp_path, erase_logs

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


def execute(cfg):
    gpu = cfg.GPU
    exp_batch = cfg.COMMON.folder
    exp_alias = cfg.COMMON.exp

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if cfg.PRELOAD_MODEL_ALIAS is not None:
        checkpoint = torch.load(
            os.path.join(
                '_logs', cfg.PRELOAD_MODEL_BATCH, cfg.PRELOAD_MODEL_ALIAS, 'checkpoints',
                str(cfg.PRELOAD_MODEL_CHECKPOINT) + '.pth'
            )
        )

    checkpoint_file = get_latest_saved_checkpoint(exp_batch, exp_alias)
    if checkpoint_file is not None:
        checkpoint = torch.load(
            os.path.join(
                '_logs', exp_batch, exp_alias, 'checkpoints', str(get_latest_saved_checkpoint(exp_batch, exp_alias))
            )
        )
        iteration = checkpoint['iteration']
        best_loss = checkpoint['best_loss']
        best_loss_iter = checkpoint['best_loss_iter']
    else:
        if not os.path.exists(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints')):
            os.mkdir(os.path.join(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints')))
        iteration = 0
        best_loss = 10000.0
        best_loss_iter = 0

    full_dataset = os.path.join(cfg.COMMON.dataset_path, cfg.TRAIN_DATASET_NAME)

    # By instantiating the augmenter we get a callable that augment images and transform them
    # into tensors.
    # augmenter = Augmenter(cfg.AUGMENTATION)
    augmenter = None

    # Instantiate the class used to read a dataset. The coil dataset generator
    # can be found
    dataset = CoILDataset(full_dataset, cfg, transform=augmenter)  # this has g_conf
    print("Loaded dataset")

    data_loader = select_balancing_strategy(dataset, cfg, iteration)
    model = CoILModule(cfg)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    if checkpoint_file is not None or cfg.PRELOAD_MODEL_ALIAS is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        accumulated_time = checkpoint['total_time']
        loss_window = checkpoint['loss_window']
    else:  # We accumulate iteration time and keep the average speed
        accumulated_time = 0
        loss_window = []

    print("Before the loss")

    criterion = Loss(cfg.LOSS_FUNCTION)

    # Loss time series window
    for data in data_loader:

        iteration += 1
        if iteration % 1000 == 0:
            adjust_learning_rate_auto(
                optimizer, loss_window, cfg.LEARNING_RATE, cfg.LEARNING_RATE_THRESHOLD, cfg.LEARNING_RATE_DECAY_LEVEL
            )

        capture_time = time.time()
        controls = data['directions']
        model.zero_grad()
        branches = model(torch.squeeze(data['rgb'].cuda()), dataset.extract_inputs(data).cuda())
        loss_function_params = {
            'branches': branches,
            'targets': dataset.extract_targets(data).cuda(),
            'controls': controls.cuda(),
            'inputs': dataset.extract_inputs(data).cuda(),
            'branch_weights': cfg.BRANCH_LOSS_WEIGHT,
            'variable_weights': cfg.VARIABLE_WEIGHT
        }
        loss, _ = criterion(loss_function_params)
        loss.backward()
        optimizer.step()

        if is_ready_to_save(iteration, cfg):
            state = {
                'iteration': iteration,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'total_time': accumulated_time,
                'optimizer': optimizer.state_dict(),
                'best_loss_iter': best_loss_iter,
                'loss_window': loss_window
            }
            torch.save(state, os.path.join('_logs', exp_batch, exp_alias, 'checkpoints', str(iteration) + '.pth'))

        if loss.data < best_loss:
            best_loss = loss.data.tolist()
            best_loss_iter = iteration

        accumulated_time += time.time() - capture_time

        loss_window.append(loss.data.tolist())
        print("Iteration: %d  Loss: %f" % (iteration, loss.data))


if __name__ == '__main__':
    cfg = EasyDict(train_config)
    create_log_folder(cfg.COMMON.folder)
    erase_logs(cfg.COMMON.folder)

    create_exp_path(cfg.COMMON.folder, cfg.COMMON.exp)

    execute(cfg)
