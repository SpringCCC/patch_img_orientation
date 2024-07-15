from .cos_sin_loss import *
from .gauss import *
from .single_angle import *


def get_loss(opt):
    loss_name = opt.loss_name
    loss_class = None
    if loss_name == 0:
        loss_class = CosSin_Loss()
    elif loss_name == 1: #gauss
        loss_class = Gauss_Loss()
    elif loss_name == 2:
        loss_class = SingleAngle_Loss()
        
    return loss_class.calc_loss
