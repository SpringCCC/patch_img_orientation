import numpy as np
import math
from springc_utils import *
import torch.nn as nn
from .abc_loss import LossAbstractClass

class Gauss_Loss(LossAbstractClass):

    def __init__(self, class_num=360, sigma=0.005) -> None:
        idx = np.arange(class_num)
        mn = np.mean(idx)
        rg = np.max(idx)
        idx = (idx-mn) / rg * 2  # 缩放到[-1, 1]之间
        pdf = np.exp(-np.power(idx,2)/sigma).astype(np.float32)
        self.pdf = np.hstack((pdf[int(class_num/2):],pdf[:int(class_num/2)]))

    def angle2label(self, ag):
    
        class_num = len(self.pdf)
        csG = 360/class_num
        idx = int(np.floor(ag/csG))
        l = len(self.pdf)
        label = np.hstack((self.pdf[l-idx:], self.pdf[:l-idx]))
        return label
    
    def label2angle(lb):
        class_num = len(lb)
        csG = 360/class_num
        idm = np.argmax(lb)
        return idm*csG + csG/2
    
    def calc_loss(self, predict, angle):
        angle = toNumpy(angle)
        degree = np.asarray([round(math.degrees(a)) for a in angle])
        degree = np.clip(degree, -180, 179)
        degree += 180
        gt_label = np.asarray([self.angle2label(d) for d in degree])
        gt_label = toTensor(gt_label).float().cuda()
        mse_loss = nn.MSELoss()
        loss = mse_loss(predict, gt_label)
        return loss

