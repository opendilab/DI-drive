import re
import os


def static_vars(**kwargs):

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def tryint(s):
    try:
        return int(s)
    except:
        return s


def command_number_to_index(command_vector):

    return command_vector - 2


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def create_log_folder(exp_batch_name):
    """
        Only the train creates the path. The validation should wait for the training anyway,
        so there is no need to create any path for the logs. That avoids race conditions.
    Returns:

    """
    # This is hardcoded the logs always stay on the _logs folder
    root_path = '_logs'

    if not os.path.exists(root_path):
        os.mkdir(root_path)

    if not os.path.exists(os.path.join(root_path, exp_batch_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name))


def create_exp_path(exp_batch_name, experiment_name):
    # This is hardcoded the logs always stay on the _logs folder
    root_path = '_logs'

    if not os.path.exists(os.path.join(root_path, exp_batch_name, experiment_name)):
        os.mkdir(os.path.join(root_path, exp_batch_name, experiment_name))


def erase_logs(exp_batch_name):

    root_path = '_logs'

    experiments = os.listdir(os.path.join(root_path, exp_batch_name))

    for exp in experiments:
        if os.path.isdir(os.path.join(root_path, exp_batch_name, exp)):
            experiments_logs = os.listdir(os.path.join(root_path, exp_batch_name, exp))
            for log in experiments_logs:
                if not os.path.isdir(os.path.join(root_path, exp_batch_name, exp, log))\
                        and '.csv' not in log:
                    os.remove(os.path.join(root_path, exp_batch_name, exp, log))
