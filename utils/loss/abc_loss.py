from abc import ABC, abstractmethod
from springc_utils import *

class LossAbstractClass(ABC):

    @abstractmethod
    def calc_loss(self):
        pass

    def convert_angle_to_degree(self):
        pass


class BasicLoss(LossAbstractClass):

    def __init__(self) -> None:
        pass
    
    # 从正常图像（-pi, pi）之间转换为（0， 360）注意，0-->360是从水平x轴负方向，顺时针旋转
    def convert_angle_to_degree(self, angle):
        angle = toNumpy(angle)
        degree = np.asarray([round(math.degrees(a)) for a in angle])
        degree = np.clip(degree, -180, 179)
        degree += 180
        return degree