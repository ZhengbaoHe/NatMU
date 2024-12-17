import sys, os
from collections import OrderedDict
import torch.nn.init as init
import numpy as np
from sklearn.decomposition import PCA
import shutil
import torch
import torch.nn as nn
from PIL import Image
import shutil
import random


sys.path.append('..')
import models
import datetime
import time
from .output import format_number
import torchvision.models as torch_models
# from torchvision.models import resnet50, ResNet50_Weights


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    
    
def saveModel(state, path):
    torch.save(state, path)
    print("Saved at {}.".format(path))


class DefaultModel(nn.Module):
    def __init__(self, net, data_normalize):
        super(DefaultModel, self).__init__()
        self.net = net
        if data_normalize is None:
            self.data_normalize = None
        else:
            self.data_normalize = data_normalize

    def forward(self, x):
        if self.data_normalize is not None:
            x = self.data_normalize(x)
        return self.net(x)
    

def copy_files(source, dst, isFile=False):
    print("Copy from {} to {}".format(source, dst))
    try:
        if isFile:
            shutil.copy(source, dst)
        else:
            shutil.copytree(source, dst, copy_function=shutil.copy)
    except Exception as e:
        print("Copy failed, error:" + str(e))
    else:
        print("Copy success")

def save_running_results(args):
    try:
        if args.debug:
            shutil.move(os.path.dirname(args.ckpt), os.path.dirname(args.ckpt).replace("tmp", "debug_results"))
        else:
            shutil.move(os.path.dirname(args.ckpt), os.path.dirname(args.ckpt).replace("tmp", "ckpt"))
    except Exception as e:
        print(e)

def load_model(net, args):
    print('==> Resuming from checkpoint..')
    print("ckpt folder:", args.resumeCKPT)
    assert os.path.exists(args.resumeCKPT), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resumeCKPT)
    net.load_state_dict(checkpoint['net'])


def update_ckpt(args):
    ckpt_tmp = args.ckpt.replace(".pth", "")
    basic_dir = f"tmp/{args.model}/{args.dataset}/{ckpt_tmp}/"
    if hasattr(args, "forget_class"):
        basic_dir += f"{args.forget_class}/"
    basic_dir += f"{datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S.%f')[:-4]}"
    if not args.debug:
        config = vars(args)
        
        for key in ["opt",]:
            if key in config:
                basic_dir += "_{}".format(config[key])
        
        for key in [ "lr", "epoch", "batchsize", "wd", "threshold", "std", "delta", "alpha", "init_linear"]:
            if key in config:
                if type(config[key]) is float:
                    basic_dir += "_{}-{:.5f}".format(key, config[key])
                else:
                    basic_dir += "_{}-{}".format(key, config[key])


    print("Save path:", basic_dir)
    time.sleep(1)
    absolute_code_path = os.path.abspath(args.code_file)
    dir = os.path.dirname(absolute_code_path)

    if not (hasattr(args, "local_rank") and args.local_rank != 0):
        # if not os.path.exists(os.path.dirname(os.path.dirname(basic_dir))):
        #     os.mkdir(os.path.dirname(os.path.dirname(basic_dir)))
            
        # if not os.path.exists(os.path.dirname(basic_dir)):
        #     os.mkdir(os.path.dirname(basic_dir))
        
        if not os.path.exists(basic_dir):
            os.makedirs(basic_dir, exist_ok=True)

        save_code_path = os.path.join(basic_dir, "code")
        if not os.path.exists(save_code_path):
            os.mkdir(save_code_path)

        copy_files(os.path.join(dir, args.code_file), os.path.join(save_code_path, args.code_file), True)
        copy_files(os.path.join(dir, "models"), os.path.join(save_code_path, "models"))
        copy_files(os.path.join(dir, "utils"), os.path.join(save_code_path, "utils"))
    
    
    args.ckpt = os.path.join(basic_dir, "model.pth")
    args.logfile = os.path.join(basic_dir, "log.log")
    if not (hasattr(args, "local_rank") and args.local_rank != 0):
        if args.debug:
            print('-' * 60 + "\n\n\t\t\tWARNING: RUNNING IN DEBUG MODE\n\n" + '-' * 60)
        for key in vars(args):
            print(key.ljust(20), ' : ', vars(args)[key])
    time.sleep(2)
    return args


def get_model(model_name, num_of_classes=10, dataset=None, Model = None):
    if Model is None:
        Model = DefaultModel
    
    if dataset == 'cifar10':
        data_normalize = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    elif dataset == 'cifar100' or dataset == 'cifar20':
        data_normalize = NormalizeByChannelMeanStd(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    elif dataset == 'imagenet':
        data_normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset == 'tinyimagenet':
        data_normalize = NormalizeByChannelMeanStd(mean=[0.4802, 0.4481, 0.3975], std=[0.2764, 0.2689, 0.2816])
    elif dataset == 'no':
        data_normalize = None
    else:
        raise NotImplementedError
    


    if "VGG" in model_name:
        return Model(models.VGG(model_name, num_classes=num_of_classes), data_normalize)
    elif model_name == 'DenseNet':
        return Model(models.densenet_cifar(num_classes=num_of_classes), data_normalize)
    
    elif model_name=="rn50":
        return Model(torch_models.resnet50(pretrained=True), data_normalize)
    
    elif model_name=="densenet_121":
        return Model(torch_models.densenet121(pretrained=True), data_normalize)
    else:
        func = getattr(models, model_name)
        return Model(func(num_classes=num_of_classes), data_normalize)
        

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



def save_fig(img, name):
    # img: (3, 32, 32)
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1,2,0))
    img = Image.fromarray((img*255).astype(np.uint8))
    img.save(name)


import pickle
def save_records(args, test_acc, forget_acc, mia):
    if not os.path.exists("records.pkl"): return
    with open("records.pkl", "rb") as f:
        d = pickle.load(f)
    key = args.code_file[:3]
    if key in ["amn", "bli", "sal"]: 
        if args.reweight:  
            key = key + "-w"
            
    d[key]["test_acc"].append(test_acc)
    d[key]["forget_acc"].append(forget_acc)
    d[key]["mia"].append(mia)
    with open("records.pkl", "wb") as f:
        pickle.dump(d, f)

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False