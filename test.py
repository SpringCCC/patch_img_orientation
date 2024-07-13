import torch
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

from datetime import date 
# 获取当前日期  
today = date.today()  
  
# 打印当前日期  
print("今天的日期是:", today)
