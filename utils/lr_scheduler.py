
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR

def get_lr_scheduler(optimizer, opt):
    # steplr
    if opt.scheduler == 'steplr':
        lr_scheduler = StepLR(optimizer, step_size=opt.step_size)
    elif opt.scheduler == 'cos':
        lr_scheduler = CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-8)
    return lr_scheduler
