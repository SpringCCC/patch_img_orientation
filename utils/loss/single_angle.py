import numpy as np
import math
from springc_utils import *
import torch.nn as nn
from .abc_loss import LossAbstractClass



class SingleAngle_Loss(LossAbstractClass):

    def __init__(self) -> None:
        pass
    
    def calc_loss(self, predict, angle):
        angle = toNumpy(angle)
        degree = np.asarray([round(math.degrees(a))+180 for a in angle])
        degree = np.clip(degree, -180, 179)
        gt_label = toTensor(degree).float().cuda()
        mse_loss = nn.MSELoss()
        loss = mse_loss(predict, gt_label)
        return loss
