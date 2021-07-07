import os

from core.utils.others.general_helper import sort_nicely
""" FUNCTIONS FOR SAVING THE CHECKPOINTS """


def is_ready_to_save(iteration, cfg):
    """ Returns if the iteration is a iteration for saving a checkpoint

    """
    if iteration in set(cfg.SAVE_SCHEDULE):
        return True
    else:
        return False


def get_latest_saved_checkpoint(exp_batch, exp_alias):
    """
        Returns the , latest checkpoint number that was saved

    """
    if os.path.exists(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints')):
        checkpoint_files = os.listdir(os.path.join('_logs', exp_batch, exp_alias, 'checkpoints'))
        if checkpoint_files == []:
            return None
        else:
            sort_nicely(checkpoint_files)
            return checkpoint_files[-1]
    else:
        return None
