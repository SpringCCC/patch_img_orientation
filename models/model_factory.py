from .resnet_model import MyResNetModel



def get_model(opt):
    if 'resnet' in opt.model_name:
        model = MyResNetModel(opt.model_name, opt.cls_num)

    return model
