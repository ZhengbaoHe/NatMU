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
    parser.add_argument('--ckpt', type=str, default='blindspot.pth')
    parser.add_argument('--resumeCKPT', type=str, )
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save_path', '-s', action='store_true')
    parser.add_argument('--gpuid', type=str, default='0')
    
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--forget_per', type=int, default=1)
    
    parser.add_argument('--temp', type=float, default=1, help="KL temperature for Blindspot")
    args = parser.parse_args()
    return args




def get_dataloader(args, num_of_classes):
    dataSet, transform_train, transform_valid, random_subset_idx = get_dataset_transform_randomIdx(args)
    
    
    retainDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_idx=random_subset_idx, isForgetSet=False)
    forgetDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_idx=random_subset_idx, isForgetSet=True)
    testDS                            = dataSet("test", transform_valid)
    
    trainDL_full = ConcatDataset(
        retainDS_inTrainSet_woDataAugment.data, 
        retainDS_inTrainSet_woDataAugment.label,
        forgetDS_inTrainSet_woDataAugment.data, 
        forgetDS_inTrainSet_woDataAugment.label,
        transform_valid, 
        transform_valid)
    
    print(f"Data 1  Shape:{trainDL_full.data1.shape}\tData 2 Shape:{trainDL_full.data2.shape}")
    trainDS_full                      = DataLoader(trainDL_full, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True) 
    retainDL_inTrainSet_woDataAugment = DataLoader(retainDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    forgetDL_inTrainSet_woDataAugment = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    testDL                            = DataLoader(testDS, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 

    return trainDS_full, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL


def blindspot_train(student_model, full_trained_teacher, unlearning_teacher, trainloader, scheduler, 
                    optimizer:optim.Optimizer, criterion, log:LogProcessBar, device, args, KL_temperature=1):
    student_model.train()
    full_trained_teacher.eval()
    unlearning_teacher.eval()
    train_loss = AverageMeter("TrainLoss")
    correct_sample = AverageMeter("CorrectSample")
    total_sample = AverageMeter("TotalSample")
    training_time = AverageMeter("TrainingTime")
    
    for batch_idx, batch_data in enumerate(trainloader):
        inputs, targets, retain_flag = batch_data[0], batch_data[1], batch_data[2]
        
        start_time = time.time()
        num_of_batch_samples = inputs.shape[0]
        inputs, targets,retain_flag = inputs.to(device), targets.to(device), retain_flag.to(device)
        retain_flag_int = (retain_flag.reshape(-1,1)).long()


        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(inputs)
            unlearn_teacher_logits = unlearning_teacher(inputs)
            f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
            u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
            overall_teacher_out =  retain_flag_int * f_teacher_out + (1 -  retain_flag_int) * u_teacher_out
        outputs = student_model(inputs)
        student_out = F.log_softmax(outputs / KL_temperature, dim=1)
        optimizer.zero_grad()
        
        loss =  F.kl_div(student_out, overall_teacher_out, reduction="batchmean")
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        train_loss.update(loss.item(), num_of_batch_samples)
        correct_sample.update(compute_correct(outputs, targets))
        total_sample.update(num_of_batch_samples)
        training_time.update(time.time() - start_time)
        
        msg = "[{}/{}] Loss:{} | Acc:{}% | {}".format(
            format_number(2, 3, training_time.avg),
            format_number(3, 2, training_time.sum),
            # datetime.datetime.fromtimestamp(time.time() + (training_time.avg*len(trainloader)+10.6)*(args.epoch -epoch)  ).strftime("Time:%H:%M"),
            format_number(1, 3, train_loss.avg),
            format_number(3, 2, 100. * correct_sample.sum / total_sample.sum),
            "Train".ljust(15),
        )
        
        if  (batch_idx == len(trainloader)-1): log.refresh(batch_idx, len(trainloader), msg)


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
    student_model = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset).to(device)
    unlearning_teacher = deepcopy(student_model)

    
    if args.resume:
        # Load checkpoint.
        load_model(student_model, args)
    full_trained_teacher = deepcopy(student_model)

    if device == 'cuda':
        cudnn.benchmark = True
    
    trainDS_full, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL = get_dataloader(args, num_of_classes=num_of_classes)
    

    if args.opt == 'sgd':
        optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = None
        raise NotImplementedError
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_scheduler(args, len(trainDS_full)), )
    criterion = nn.CrossEntropyLoss()

    epoch_time = AverageMeter("Epoch Time")
    for epoch in range(args.epoch):
        start_time = time.time()
        log.log_print("\nEpoch:{} | Model:{} | lr now:{:.6f}".format(epoch, args.model, optimizer.state_dict()['param_groups'][0]['lr']))

        blindspot_train(student_model, full_trained_teacher, unlearning_teacher, trainDS_full, scheduler, 
                    optimizer, criterion, log, device, args, KL_temperature=args.temp)

        

        epoch_time.update(time.time() - start_time)
        print("Finished at:" + datetime.datetime.fromtimestamp(time.time() + epoch_time.val[-1]*(args.epoch -epoch)  ).strftime("Time:%H:%M"),)
    _, test_acc = valid_test(student_model, testDL, "Test Acc", device, criterion, log, args=args)
    _, retain_acc = valid_test(student_model, retainDL_inTrainSet_woDataAugment, "Retain Acc", device, criterion, log, args=args)
    _, forget_acc = valid_test(student_model, forgetDL_inTrainSet_woDataAugment, "Forget Acc", device, criterion, log, args=args)
    student_model.eval()
    
    # saveModel({'net': deepcopy(student_model.state_dict()),}, args.ckpt.replace('.pth', '-last.pth'))
    
    mia_ratio = get_membership_attack_prob(retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL, student_model)
    log.log_print("Test acc:{:.3f}\tRetain_acc:{:.3f}\tForget acc:{:.3f}\tMIA:{:.3f}".format(test_acc,retain_acc, forget_acc, mia_ratio, ))
    
    save_running_results(args)
    
    return test_acc, retain_acc, forget_acc, mia_ratio
    

    
if __name__ == "__main__":
    args = get_args()
    args.code_file = __file__
    set_resumeCKPT(args)
    args = update_ckpt(args)
    main(args)
    