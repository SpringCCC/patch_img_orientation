'''
@File    :   preprocess.py
@Time    :   2024/07/12 09:49:01
@Author  :   SpringC
@Version :   1.0
@Contact :   huangw@xy.gabjoy.com
@Desc    :   传入opencv读取的原始rgb图像，在这里进行预处理，由于可能需要处理不同
'''
import numpy as np
import cv2
from abc import ABC, abstractmethod
from config import Config
from springc_utils import *
from torchvision import transforms
from PIL import Image
import random

class AbstractProcess(ABC):

    @abstractmethod
    def build_transform(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

class BasicProcess(AbstractProcess):

    def __init__(self, opt:Config, is_Train):
        self.opt = opt
        self.pad_value = self.opt.pad_value
        self.is_Train = is_Train

    def pad(self, img):
        height, width, _ = img.shape
        if height > width:  
            # 若高度大于宽度，则在左右两侧填充  
            padding_size = height - width
            padding_size = random.randint(0, padding_size)
            if self.opt.center_pad:
                padding_size = (height - width) // 2
            padding = (0, 0, padding_size, height - width - padding_size)
        else:  
            # 若宽度大于或等于高度，则在上下两侧填充  
            padding_size = width - height
            padding_size = random.randint(0, padding_size)
            if self.opt.center_pad:
                padding_size = (height - width) // 2
            padding = (padding_size, width - height - padding_size, 0, 0)    
        padded_image = cv2.copyMakeBorder(img, *padding, cv2.BORDER_CONSTANT, value=[self.pad_value, self.pad_value, self.pad_value])
        return padded_image
    
    def normalize(self, img):
        # normalize_type: 
        #   0:  (-1, 1)
        #   1:  (0, 1)
        # 在transforms.ToTensor()之后使用 img本身就是[0, 1]之间
        if self.opt.normalize_type == 0:
            img *= 2
            img -= 1
        elif self.opt.normalize_type == 1:
            pass
        else:
            raise ValueError("未定义的图像归一化方法")
        return img
    
    def invert_normalize(self, img):
        img = toNumpy(img)
        if self.opt.normalize_type == 0:
            img += 1
            img /= 2
            img *= 255
            img = int(img)
            img = np.clip(img, 0, 255)
        elif self.opt.normalize_type == 1:
            img *= 255
            img = int(img)
            img = np.clip(img, 0, 255)
        else:
            raise ValueError("未定义的图像归一化方法")
        return img
    
    def build_transform(self):
        """
        图像增强方式：
            多尺寸+中心裁剪+亮度调整+对比度调整+色彩抖动+色调调整
        """
        # todo: pytorch版本低，导致torchvision版本也低，后续换新环境，增加新的数据增强
        if self.is_Train:
            tsf = []
            if random.random() > 0.5:
                tsf.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            if random.random() > 0.8:
                tsf.append(transforms.GaussianBlur(random.randint(1, 10)*2 + 1, random.randint(1, 10)))
            transform = transforms.Compose([
                transforms.Resize((self.opt.img_size, self.opt.img_size)),
                *tsf,
                transforms.ToTensor(),
                ])
        else:
            transform = transforms.Compose([
                # transforms.Resize((self.opt.random_img_size, self.opt.random_img_size)),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩抖动  
                # transforms.AdjustBrightness(brightness_factor=0.2),  # 亮度调整，参数为亮度因子  
                # transforms.AdjustContrast(contrast_factor=0.2),  # 对比度调整，参数为对比度因子  
                # transforms.AdjustHue(hue_factor=0.1),  # 色调调整，参数为色调因子  
                transforms.ToTensor(),  # 转换为Tensor  
                ])
        return transform

    def augment(self, img):
        self.tsf = self.build_transform()
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = self.tsf(img)
        return img
    
    def preprocess(self, img):
        img = self.pad(toNumpy(img))
        img = self.augment(img)
        img = self.normalize(img)
        return img
    