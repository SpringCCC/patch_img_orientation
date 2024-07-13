'''
@File    :   cos_sin_loss.py
@Time    :   2024/07/12 11:21:48
@Author  :   SpringC
@Version :   1.0
@Contact :   huangw@xy.gabjoy.com
@Desc    :   按照cos sin的方式 求解预测，由于我们场景下，很多角度是竖直的，因此可以把传入的angle，做一些变换，来进行预测？有必要吗？
'''
import torch
import torch.nn as nn

def convert_angle_to_cos_sin(angles):
    cos = torch.cos(angles)
    sin = torch.cos(angles)
    gt_cos_sin = torch.stack([cos, sin], dim=1)
    return gt_cos_sin


def calc_cos_sin_loss(predict, angles):
    # predict:N, 2
    # angles: N
    gt_cos_sin = convert_angle_to_cos_sin(angles)
    mse_loss = nn.MSELoss()
    loss = mse_loss(predict, gt_cos_sin)
    return loss

