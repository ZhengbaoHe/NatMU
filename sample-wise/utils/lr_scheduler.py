import math
from functools import partial

def get_lr_scheduler(args, length):
    if args.lr_scheduler == "cosine":
        if hasattr(args, "lr_min"):
            lr_min = args.lr_min
        else:
            lr_min=0.0
        return partial(lr_lambda_cosine, 
                       warmup_epoch=args.warmup_epoch, 
                       epoch=args.epoch, lr_min=lr_min, 
                       constant_epoch=0., 
                       max_lr=args.lr, 
                       loader_length=length) 
    if args.lr_scheduler == "cosine_step": 
        return partial(lr_lambda_cosine, 
                       warmup_epoch=args.warmup_epoch, 
                       epoch=args.epoch, lr_min=0.002, 
                       constant_epoch=1., 
                       max_lr=args.lr, 
                       loader_length=length) 
    elif args.lr_scheduler == "step": 
        return partial(lr_lambda_step, 
                       warmup_epoch=args.warmup_epoch, 
                       epoch=args.epoch, lr_min=0., 
                       constant_epoch=0., 
                       max_lr=args.lr, 
                       loader_length=length)
    elif args.lr_scheduler == "constant": 
        return partial(lr_lambda_constant, 
                       warmup_epoch=args.warmup_epoch, 
                       epoch=args.epoch, lr_min=0., 
                       constant_epoch=0., 
                       max_lr=args.lr, 
                       loader_length=length)
    else: raise ValueError    

def lr_lambda_cosine(x, warmup_epoch, epoch, constant_epoch=0, lr_min=0, max_lr=0.1, loader_length=0):
    total_steps = epoch * loader_length
    warmup_steps =  warmup_epoch * loader_length
    constant_steps = constant_epoch * loader_length
    step = x+1
    if step<warmup_steps: #warmup
        return step/warmup_steps
    elif step<= total_steps -constant_steps:
        cos_value =  math.cos((step - warmup_steps) / (total_steps -warmup_steps-constant_steps) * math.pi)/2 + 0.5
        return  (cos_value * (max_lr -lr_min) + lr_min) / max_lr
    else:
        return lr_min / max_lr
    

def lr_lambda_step(x, warmup_epoch, epoch, constant_epoch=0, lr_min=0, max_lr=0.1, loader_length=0):
    total_steps = epoch * loader_length
    warmup_steps =  warmup_epoch * loader_length
    constant_steps = constant_epoch * loader_length
    step = x
    if step<warmup_steps: #warmup
        return step/warmup_steps
    elif step<= int( total_steps * .6) :
        return  (1 * (max_lr -lr_min) + lr_min) / max_lr
    elif step<= int( total_steps * .8) :
        return  (0.1 * (max_lr -lr_min) + lr_min) / max_lr
    else:
        return  (0.01 * (max_lr -lr_min) + lr_min) / max_lr
    
def lr_lambda_constant(x, warmup_epoch, epoch, constant_epoch=0, lr_min=0, max_lr=0.1, loader_length=0):
    total_steps = epoch * loader_length
    warmup_steps =  warmup_epoch * loader_length
    constant_steps = constant_epoch * loader_length
    step = x
    if step<warmup_steps: #warmup
        return step/warmup_steps
    else:
        return  1