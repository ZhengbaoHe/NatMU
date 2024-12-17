import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
from utils import *
import os
import math
import time, datetime
from functools import partial
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt

best_model = None
last_model = None

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='ResNet18')
    
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=float, default=2)
    
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--lr_scheduler', type=str, default="step")
    parser.add_argument('--wd', type=float, default=5e-4)
    
    
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--debug', '-nd', action='store_false')
    parser.add_argument('--ckpt', type=str, default='retrain.pth')
    parser.add_argument('--resumeCKPT', type=str, )
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save_path', '-s', action='store_true')
    parser.add_argument('--gpuid', type=str, default='0')
    
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar100')
    
    parser.add_argument('--forget_class', type=str)

    args = parser.parse_args()
    return args

def get_dataloader(args,):
    
    dataSet, transform_train, transform_valid, forget_class_int = get_dataset_transforms(args)
    
    retainDS_inTrainSet               = dataSet(mode="train", transform=transform_train, forget_class=forget_class_int, isForgetSet=False, )
    retainDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_class=forget_class_int, isForgetSet=False)
    forgetDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_class=forget_class_int, isForgetSet=True)
    retainDS_inTestSet                = dataSet("test", transform=transform_valid, forget_class=forget_class_int, isForgetSet=False)
    forgetDS_inTestSet                = dataSet("test", transform=transform_valid, forget_class=forget_class_int, isForgetSet=True)
    testDS                            = dataSet("test", transform=transform_valid)
    
    retainDL_inTrainSet               = DataLoader(retainDS_inTrainSet, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True) 
    retainDL_inTrainSet_woDataAugment = DataLoader(retainDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    forgetDL_inTrainSet_woDataAugment = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    retainDL_inTestSet                = DataLoader(retainDS_inTestSet, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    forgetDL_inTestSet                = DataLoader(forgetDS_inTestSet, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    testDL                            = DataLoader(testDS, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 

    return retainDL_inTrainSet, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, retainDL_inTestSet, forgetDL_inTestSet, testDL




def test(epoch, net, testloader, device, criterion, best_acc,log:LogProcessBar, best_model, last_model, args, msg="Test", save=True, forward=None):
    loss, acc,  = valid_test(net, testloader, msg, device, criterion, log, forward=forward, args=args)

    if save:
        state = {
            'net': deepcopy(net.state_dict()),
            'acc': acc,
            'epoch': epoch,
        }
        path = args.ckpt
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        last_model = state
        
        if acc > best_acc:
            best_acc = acc
            best_model = state
            
        if args.save_path:
            path = args.ckpt
            path = path.replace('.pth', '/{}.pth'.format(epoch))
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))
            saveModel(state, path)
    else:
        if acc > best_acc:
            best_acc = acc
    return acc, best_acc, best_model, last_model


def main(args=None):
    if args is None:
        args = get_args()
        args.code_file = __file__
        set_resumeCKPT(args)
        args = update_ckpt(args)
        
    log = LogProcessBar(args.logfile, args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    seed_everything(args.seed)


    if args.dataset == 'cifar10':
        num_of_classes = 10
    elif args.dataset == 'cifar100':
        num_of_classes = 100
    elif args.dataset == 'cifar20':
        num_of_classes = 20
    elif args.dataset == 'tinyimagenet':
        num_of_classes = 200
    else:
        num_of_classes = 0
        raise NotImplementedError

    print('==> Building model..')
    net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset)
    net = net.to(device)
    
    last_model = None
    start_epoch = 0  
    
    if args.resume:
        load_model(net, args)

    if device == 'cuda':
        cudnn.benchmark = True

    
    tmp_dataloaders = get_dataloader(args)
    for dataloader in tmp_dataloaders:
        log.log_print(get_dataset_info(dataloader))
        
    retainDL_inTrainSet, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment,\
        retainDL_inTestSet, forgetDL_inTestSet, testDL = tmp_dataloaders
    
    
    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = None
        raise NotImplementedError
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_scheduler(args, len(retainDL_inTrainSet)), )
    criterion = nn.CrossEntropyLoss()

    epoch_time = AverageMeter("Epoch Time")
    for epoch in range(start_epoch, args.epoch):
        start_time = time.time()
        log.log_print("\nEpoch:{} | Model:{} | lr now:{:.6f}".format(epoch, args.model, optimizer.state_dict()['param_groups'][0]['lr']))

        train_vanilla(net, retainDL_inTrainSet, scheduler, optimizer, criterion, log, device, args)
        _, test_acc = valid_test(net, retainDL_inTestSet, "Test Acc on Test", device, criterion, log, args=args)
        _, retain_acc = valid_test(net, retainDL_inTrainSet_woDataAugment, "Retain Acc on Train", device, criterion, log, args=args)
        _, forget_acc_on_train = valid_test(net, forgetDL_inTrainSet_woDataAugment, "Forget Acc on Train", device, criterion, log, args=args)
        _, forget_acc_on_test = valid_test(net, forgetDL_inTestSet, "Forget Acc on Test", device, criterion, log, args=args)
        
        epoch_time.update(time.time() - start_time)
        print("Finished at:" + datetime.datetime.fromtimestamp(time.time() + epoch_time.val[-1]*(args.epoch -epoch)  ).strftime("Time:%H:%M"),)

    
    mia_ratio = get_membership_attack_prob(retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL, net)
    log.log_print(f"Test Acc:{test_acc:.3f}\tRetain Acc:{retain_acc:.3f}Forget Acc on Train:{forget_acc_on_train:.3f}\tForget Acc on Test:{forget_acc_on_test:.3f}\tMIA:{mia_ratio:.3f}")
    

    
    last_model = {
            'net': deepcopy(net.state_dict()),
            'epoch': epoch,
        }
    saveModel(last_model, args.ckpt.replace('.pth', '-last.pth'))    
    save_running_results(args)
    
if __name__ == "__main__":
    main()