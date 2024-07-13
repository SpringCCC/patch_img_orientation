from springc_utils import *
from config import Config
from PIL import Image
from load_data.process import BasicProcess

class Dataset:

    def __init__(self, opt:Config, is_Train=True) -> None:
        self.opt = opt
        self.is_Train = is_Train
        if self.is_Train:
            self.imgs_path, self.angles = self.readtxt(opt.train_txt)
        else:
            self.imgs_path, self.angles = self.readtxt(opt.val_txt)
        self.filter_img_size() #过滤掉图像尺寸不合要求的图像
        self.process = BasicProcess(opt, self.is_Train)

    def filter_img_size(self):
        imgs_path, angles = [], []
        for img_path, angle in zip(self.imgs_path, self.angles):
            w, h = Image.open(img_path).size
            if max(w, h) > self.opt.filter_size_max or min(w, h) < self.opt.filter_size_min:continue
            imgs_path.append(img_path)
            angles.append(angle)
        self.imgs_path, self.angles = imgs_path, angles

    def __getitem__(self, idx):
        img_path, angle = self.imgs_path[idx], self.angles[idx]
        img = read_img(img_path)
        img = self.process.preprocess(img)
        return img, angle

    def __len__(self):
        return len(self.angles)
    
    def readtxt(self, txt_path):
        imgs_path = []
        angles = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imgs_path.append(line[0])
                angles.append(float(line[1]))
        return imgs_path, angles
