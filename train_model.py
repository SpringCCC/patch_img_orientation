'''
@File    :   train_model.py
@Time    :   2024/07/12 11:09:37
@Author  :   SpringC
@Version :   1.0
@Contact :   huangw@xy.gabjoy.com
@Desc    :   当前文件作用
'''
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
from torch.utils.tensorboard import SummaryWriter
import fire

def loginfo(opt:Config):
    for k, v in opt.__dict__.items():
        if not k.startswith("_"):
            logging.info(f"{k}:{v}")
    logging.info(f"加载 {opt.model_name} 成功...")
    logging.info(f"加载训练集 {opt.train_txt} 成功...")
    logging.info(f"加载验证   {opt.val_txt} 成功...")
    logging.info(f"加载损失函数 {opt.loss_name} 成功...")
    logging.info(f"加载优化器 {opt.optimizer} 成功...")
    logging.info(f"加载调度器 {opt.scheduler} 成功...")
    logging.info(f"准备训练模型...")

def build_dataloader(opt):
    train_dataset = Dataset(opt, is_Train=True)
    val_dataset   = Dataset(opt, is_Train=False)
    train_dataloader = DataLoader(train_dataset, opt.bs, num_workers=opt.num_workers, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(val_dataset, opt.val_bs, num_workers=1, shuffle=False)
    return train_dataloader, val_dataloader

def train(opt:Config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda)
    setup_logging(os.path.join(os.path.dirname(__file__), f'run/log/', opt.log_dir_name+".log"))
    logging.info("*"*100)
    writer = SummaryWriter(log_dir=opt.log_dir)
    model = get_model(opt)
    if opt.model_weight:
        model = model.load_state_dict(torch.load(opt.model_weight))
    model.cuda()
    train_dataloader, val_dataloader = build_dataloader(opt)
    criterion = get_loss(opt)
    if opt.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=opt.lr)
    scheduler = get_lr_scheduler(optimizer, opt)
    loginfo(opt)
    best_val_loss = 1e5
    for epoch in tqdm(range(opt.epochs)):
        train_one_epoch(model, train_dataloader, optimizer, criterion, epoch, writer)
        if not opt.is_fix_lr:
            scheduler.step()
        avg_loss = val(model, val_dataloader, criterion, epoch, best_val_loss, writer)
        best_val_loss = avg_loss if avg_loss<best_val_loss else best_val_loss

        
@timing_decorator(logging)
def train_one_epoch(model, train_dataloader, optimizer, criterion, epoch, writer):
    model.train()
    running_loss = 0
    for img, angle in tqdm(train_dataloader):
        optimizer.zero_grad()
        img = img.float().cuda()
        angle = angle.float().cuda()
        predict = model(img)
        loss = criterion(predict, angle)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=opt.clip_value) # 梯度裁剪
        optimizer.step()
        running_loss += loss.item()
        # opt.img_size = random.choices(opt.random_img_size,weights=opt.random_weight, k=1)[0]# 每一个step,使用不同尺寸图像训练
        opt.img_size = 416 # froze train size
    avg_loss = running_loss / len(train_dataloader)
    writer.add_scalar("training_loss", avg_loss, epoch)
    writer.add_scalar("lr", optimizer.state_dict()['param_groups'][0]['lr'], epoch)

@timing_decorator(logging)
def val(model, val_dataloader, criterion, epoch, best_val_loss, writer):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for img, angle in tqdm(val_dataloader):
            img = img.float().cuda()
            angle = angle.float().cuda()
            predict = model(img)
            loss = criterion(predict, angle)
            running_loss += loss.item()
    avg_loss = running_loss / len(val_dataloader)
    writer.add_scalar("val_loss", avg_loss, epoch)
    #保存模型
    if avg_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(opt.sv_model_path, 'best.pth'))
        logging.info(f"epoch:{epoch} best_val_loss:{best_val_loss}, avg_loss:{avg_loss}, 保存第{epoch}个模型为最好模型")
    else:
        logging.info(f"epoch:{epoch} best_val_loss:{best_val_loss}, avg_loss:{avg_loss}, ")
    return avg_loss

def main(**kwargs):
    opt._parse(kwargs)
    train(opt)


if __name__ == '__main__':
    debug = True
    if debug:
        opt._parse({})
        train(opt)
    else:
        fire.Fire()