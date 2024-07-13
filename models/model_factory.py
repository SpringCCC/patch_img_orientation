from .resnet_model import MyResNetModel



def get_model(model_name):
    if 'resnet' in model_name:
        model = MyResNetModel(model_name)

    return model
