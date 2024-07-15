import numpy as np
import math
from springc_utils import *
import torch.nn as nn
from .abc_loss import BasicLoss



class SingleAngle_Loss(BasicLoss):

    def __init__(self) -> None:
        pass
    
    def calc_loss(self, predict, angle):
        degree = self.convert_angle_to_degree(angle)
        gt_label = toTensor(degree).float().cuda()
        mse_loss = nn.MSELoss()
        loss = mse_loss(predict, gt_label.reshape(-1, 1))
        return loss
