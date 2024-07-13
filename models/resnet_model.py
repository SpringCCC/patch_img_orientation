import torch.nn as nn


import torch  
import torch.nn as nn  
from torchvision.models import resnet34  
  
class MyResNetModel(nn.Module):

    def __init__(self, model_name):  
        super(MyResNetModel, self).__init__()
        if '34' in model_name:  
            self.resnet = resnet34(pretrained=True)  
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  
  
    def forward(self, x):  
        x = self.resnet(x)
        return x  