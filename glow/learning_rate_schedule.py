import numpy as np


def default(init_lr, global_step):
    return init_lr


# https://github.com/tensorflow/tensor2tensor/issues/280#issuecomment-339110329
def noam_learning_rate_decay(init_lr, global_step, warmup_steps=4000, minimum=None):
     # Noam scheme from tensor2tensor:
    #print('init_lr', init_lr)    
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    if minimum is not None and global_step > warmup_steps:
        if lr < minimum:
            lr = minimum
    return lr


def step_learning_rate_decay(init_lr, global_step,
                             anneal_rate=0.98,
                             anneal_interval=30000):
    return init_lr * anneal_rate ** (global_step // anneal_interval)


def cyclic_cosine_annealing(init_lr, global_step, T, M):
    """Cyclic cosine annealing

    https://arxiv.org/pdf/1704.00109.pdf

    Args:
        init_lr (float): Initial learning rate
        global_step (int): Current iteration number
        T (int): Total iteration number (i,e. nepoch)
        M (int): Number of ensembles we want

    Returns:
        float: Annealed learning rate
    """
    TdivM = T // M
    return init_lr / 2.0 * (np.cos(np.pi * ((global_step - 1) % TdivM) / TdivM) + 1.0)
