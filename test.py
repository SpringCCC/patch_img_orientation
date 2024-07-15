import os
import torch
from config import Config, opt
from springc_utils import *
from tqdm import tqdm
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from models.model_factory import get_model
from load_data.dataset import Dataset
from utils.loss.loss_factory import get_loss
from utils.lr_scheduler import get_lr_scheduler
from torch.nn.utils import clip_grad_norm_
from analyze.tool import *

def test(opt:Config):
    opt._parse({})
    model_weight = opt.model_weight
    model = get_model(opt.model_name)
    model.load_state_dict(torch.load(model_weight))
    model.cuda()
    opt.val_txt = opt.test_txt
    test_dataset   = Dataset(opt, is_Train=False)
    test_dataloader = DataLoader(test_dataset, opt.val_bs, num_workers=1, shuffle=False)
    model.eval()
    pred_degrees, gt_degress = [], []
    with torch.no_grad():
        for img, angle in tqdm(test_dataloader):
            img = img.float().cuda()
            angle = angle.float().cuda()
            predict = model(img)
            pred_degree = convert_cossin_to_angle(predict)
            gt_degree = [round(math.degrees(a)) for a in angle]
            pred_degrees.extend(pred_degree)
            gt_degress.extend(gt_degree)
    diff_degrees = [p-g for p, g in zip(pred_degrees, gt_degress)]
    diff_degrees = [abs(d) for d in diff_degrees]
    diff_degrees = [min(360-d, d) for d in diff_degrees]
    show_diff_angle_per1(gt_degress, diff_degrees)


if __name__ == '__main__':
    test(opt)


