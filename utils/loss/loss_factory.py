from .cos_sin_loss import *

def get_loss(loss_name):
    if loss_name == 0:
        loss = calc_cos_sin_loss

    return loss
