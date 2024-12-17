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
    
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--warmup_epoch', type=float, default=0)
    
    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--lr_scheduler', type=str, default="cosine")
    parser.add_argument('--wd', type=float, default=0)
    
    
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--debug', '-nd', action='store_false')
    parser.add_argument('--ckpt', type=str, default='amnesiac.pth')
    parser.add_argument('--resumeCKPT', type=str, )
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save_path', '-s', action='store_true')
    parser.add_argument('--gpuid', type=str, default='0')
    
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--forget_per', type=int, default=1)
    
    args = parser.parse_args()
    return args




def get_dataloader(args, num_of_classes):
    dataSet, transform_train, transform_valid, random_subset_idx = get_dataset_transform_randomIdx(args)
    if "cifar" in args.dataset:
        transform_train = transform_train_cifar_patch
    
    
    retainDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_idx=random_subset_idx, isForgetSet=False)
    forgetDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_idx=random_subset_idx, isForgetSet=True)
    testDS                            = dataSet("test", transform_valid)
    

    random_idx_of_retainData = get_random_idx_of_retainData(retain_labels=retainDS_inTrainSet_woDataAugment.label.copy(), 
                                 forget_labels=forgetDS_inTrainSet_woDataAugment.label.copy(), 
                                 pattern_length=1, 
                                 num_of_classes=num_of_classes)
        
    random_label = retainDS_inTrainSet_woDataAugment.label.copy()[random_idx_of_retainData]
    
    trainDS_with_randomLabel = ConcatDataset(
        retainDS_inTrainSet_woDataAugment.data, 
        retainDS_inTrainSet_woDataAugment.label,
        forgetDS_inTrainSet_woDataAugment.data, 
        random_label,
        transform_valid, 
        transform_valid)
    
    print(f"Data 1  Shape:{trainDS_with_randomLabel.data1.shape}\tData 2 Shape:{trainDS_with_randomLabel.data2.shape}")
    trainDL_with_randomLabel          = DataLoader(trainDS_with_randomLabel, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True) 
    retainDL_inTrainSet_woDataAugment = DataLoader(retainDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    forgetDL_inTrainSet_woDataAugment = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    testDL                            = DataLoader(testDS, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 

    return trainDL_with_randomLabel, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL



def main(args=None):
    if args is None:
        args = get_args()
        args.code_file = __file__
        set_resumeCKPT(args)
        args = update_ckpt(args)
    print_args(args)
    os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.logfile), "code"), exist_ok=True)
    log = LogProcessBar(args.logfile, args)
    seed_everything(args.seed)
    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

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
    net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset).to(device)

    
    if args.resume:
        # Load checkpoint.
        load_model(net, args)

    if device == 'cuda':
        cudnn.benchmark = True
    
    trainDL_with_randomLabel, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL = get_dataloader(args, num_of_classes=num_of_classes)
    
    
    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = None
        raise NotImplementedError
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_scheduler(args, len(trainDL_with_randomLabel)), )
    criterion = nn.CrossEntropyLoss()
    epoch_time = AverageMeter("Epoch Time")

    for epoch in range(args.epoch):
        start_time = time.time()
        log.log_print("\nEpoch:{} | Model:{} | lr now:{:.6f}".format(epoch, args.model, optimizer.state_dict()['param_groups'][0]['lr']))

        train_vanilla(net, trainDL_with_randomLabel, scheduler, optimizer, criterion, log, device, args,)

        

        epoch_time.update(time.time() - start_time)
        print("Finished at:" + datetime.datetime.fromtimestamp(time.time() + epoch_time.val[-1]*(args.epoch -epoch)  ).strftime("Time:%H:%M"),)
    _, test_acc = valid_test(net, testDL, "Test Acc", device, criterion, log, args=args)
    _, retain_acc = valid_test(net, retainDL_inTrainSet_woDataAugment, "Retain Acc", device, criterion, log, args=args)
    _, forget_acc = valid_test(net, forgetDL_inTrainSet_woDataAugment, "Forget Acc", device, criterion, log, args=args)
    net.eval()

    # saveModel({'net': deepcopy(net.state_dict()),}, args.ckpt.replace('.pth', '-last.pth'))
    mia_ratio = get_membership_attack_prob(retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL, net)
    log.log_print("Test acc:{:.3f}\tRetain_acc:{:.3f}\tForget acc:{:.3f}\tMIA:{:.3f}".format(test_acc,retain_acc, forget_acc, mia_ratio, ))

    save_running_results(args)
    
    return test_acc, retain_acc, forget_acc, mia_ratio


if __name__ == "__main__":
    args = get_args()
    args.code_file = __file__
    set_resumeCKPT(args)
    args = update_ckpt(args)
    main(args)
    