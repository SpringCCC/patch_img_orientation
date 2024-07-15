import os
from dataclasses import dataclass
from pprint import pprint
from datetime import date

@dataclass
class Config():

# python train_model.py main --model_name='resnet34' --bs=192 --val_bs=192 --loss_name=1 --optimizer='adam' --scheduler='cos' --model_weight='' --epochs=150 --normalize_type=0 --is_fix_lr=True --cuda='0' --data_index=1
    

    #动态生成数据集
    debug = False
    train_date = date.today()
    txt_root = r"/mnt/hd1/springc/code/work/vehicle_orientation/patch_img_orientation/assets/"
    base_sv_model_path = f"/mnt/hd1/springc/train_model_save/nuscene_car_direct/{train_date}/"

    train_txt = None          #占位符
    val_txt = None
    test_txt = None
    model_weight = None
    step_size = 1
    clip_value = 10 #梯度裁剪值
    random_img_size = [64, 96, 128, 224, 256, 384, 512, 640]
    random_weight = [3, 5, 5, 3, 3, 2, 1, 1]
    
    # 过滤掉不符合要求的图像尺寸
    filter_size_max = 800
    filter_size_min = 10

    # model_weight = r"/mnt/hd1/springc/train_model_save/nuscene_car_direct/2024-07-13/resnet34_0_adam_800_0_cos_data1/best.pth"
    #需要修改的參數
    data_index = 1            # 可选项[1,2,3,4,5,6]  ********************
    bs = 128                  # train時使用192
    val_bs = 128              # train時使用192
    lr = 1e-3
    is_fix_lr = True          #前期测试用，为True时，表示不管设置什么lr_scheduler,都是固定的lr
    scheduler = 'steplr'      # cos ********************
    epochs = 100           
    num_workers = 4
    pad_value = 0
    normalize_type = 0 # 0-->(-1, 1)   1--->(0, 1) #  ********************
    img_size = 416 #多尺寸训练，可选项[32, 64, 96, 128, 224, 256, 384, 512, 640]
    center_pad = False        # 图像pad成方形，是否中心pad还是随机位置pad

    cls_num = 1   
    loss_name_dict={0:'cos_sin', 1:'gauss', 2:'single_angle'}
    loss_name = 2  #损失函数类别 可选项['cos_sin', 'gauss', 'single_angle']
    cuda = '1'

    model_name = 'resnet34' #  ********************
    optimizer = 'adam' # 'sgd' ********************

    def _parse(self, kwargs:dict):
        if self.debug:
            self.base_log_dir = "run/debug/experiment" 
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError(f"Unknown Options: --{k}")
            setattr(self, k, v)
        assert self.data_index>0 and self.data_index<7, print(f"请输入训练数据集序号，可选项:[1,2,3,4,5,6]")
        self.train_txt = os.path.join(self.txt_root, f"train_{self.data_index}.txt")
        self.val_txt   = os.path.join(self.txt_root, f"val_{self.data_index}.txt")
        self.test_txt  = os.path.join(self.txt_root, f"test_{self.data_index}.txt")
        self.log_dir_name = f"{self.model_name}_{self.loss_name_dict[self.loss_name]}_{self.optimizer}_fixlr{int(self.is_fix_lr)}_{self.filter_size_max}_{self.normalize_type}_{self.scheduler}_data{self.data_index}"
        self.sv_model_path = os.path.join(self.base_sv_model_path, self.log_dir_name)
        os.makedirs(self.sv_model_path, exist_ok=True)
        self.log_dir = os.path.join(self.base_log_dir, self.log_dir_name)
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k:getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith("_")}
    

opt = Config()



