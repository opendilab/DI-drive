import numpy as np
import scipy.stats


def decreasing_probability(values: np.ndarray) -> float:

    n_steps = len(values)
    steps = np.arange(n_steps)
    A = np.vstack([steps, np.ones(n_steps)]).T
    loc, shift = np.linalg.lstsq(A, values, rcond=None)[0]
    values_ = loc * steps + shift
    sigma2 = np.sum((values - values_) ** 2) / (n_steps - 2)
    scale = np.sqrt(12 * sigma2 / (n_steps ** 3 - n_steps))
    return scipy.stats.norm.cdf(0., loc=loc, scale=scale)


def steps_without_decrease(values: np.ndarray, robust: bool = False) -> int:

    if robust:
        values = np.array(values)[values < np.percentile(values, 90)]
    steps_without_decrease = 0
    n_steps = len(values)
    for i in reversed(range(n_steps)):
        p = decreasing_probability(values[i:])
        if p < 0.51:
            steps_without_decrease = n_steps - i
    return steps_without_decrease


def adjust_learning_rate(optimizer, num_iters, LEARNING_RATE, LEARNING_RATE_DECAY_INTERVAL, LEARNING_RATE_DECAY_LEVEL):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    cur_iters = num_iters
    minlr = 0.0000001
    scheduler = "normal"
    learning_rate = LEARNING_RATE
    decayinterval = LEARNING_RATE_DECAY_INTERVAL
    decaylevel = LEARNING_RATE_DECAY_LEVEL
    if scheduler == "normal":
        while cur_iters >= decayinterval:
            learning_rate = learning_rate * decaylevel
            cur_iters = cur_iters - decayinterval
        learning_rate = max(learning_rate, minlr)

    for param_group in optimizer.param_groups:
        print("New Learning rate is ", learning_rate)
        param_group['lr'] = learning_rate


def adjust_learning_rate_auto(
    optimizer, loss_window, LEARNING_RATE, LEARNING_RATE_THRESHOLD, LEARNING_RATE_DECAY_LEVEL
):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    minlr = 0.0000001
    learning_rate = LEARNING_RATE
    thresh = LEARNING_RATE_THRESHOLD
    decaylevel = LEARNING_RATE_DECAY_LEVEL
    n = 1000
    start_point = 0
    print("Loss window ", loss_window)
    while n < len(loss_window):
        print("Startpoint ", start_point, " N  ", n)
        steps_no_decrease = steps_without_decrease(loss_window[start_point:n])
        steps_no_decrease_robust = steps_without_decrease(loss_window[start_point:n], robust=True)
        print("no decrease, ", steps_no_decrease, " robust", steps_no_decrease_robust)
        if steps_no_decrease > thresh and steps_no_decrease_robust > thresh:
            start_point = n
            learning_rate = learning_rate * decaylevel

        n += 1000

    learning_rate = max(learning_rate, minlr)

    for param_group in optimizer.param_groups:
        print("New Learning rate is ", learning_rate)
        param_group['lr'] = learning_rate
