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
import wandb

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
    parser.add_argument('--ckpt', type=str, default='natmu.pth')
    parser.add_argument('--resumeCKPT', type=str, )
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save_path', '-s', action='store_true')
    parser.add_argument('--gpuid', type=str, default='0')
    
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--forget_per', type=int, default=1)

    args = parser.parse_args()
    return args



@torch.no_grad()
def get_random_idx(forget_loader, net, pattern_length, device, retainLabel,num_of_classes):
    # 为了防止某一类的样本被挑选完，导致报错
    # Due to the possibility that the top-n predicted confidence categories of certain samples may belong to the same class, this function prevents errors caused by the depletion of samples from a particular class.
    # 
    net.eval()
    idx_of_retain_samples_per_class = []
    for class_idx in range(num_of_classes):
        idx_of_retain_samples_per_class.append(np.nonzero(retainLabel==class_idx)[0])
    for class_idx in range(num_of_classes):
        np.random.shuffle(idx_of_retain_samples_per_class[class_idx])
    top_k = []
    idx = []
    for x,y in forget_loader:
        print(len(idx))
        x, y = x.to(device), y.to(device)
        outputs = net(x)
        mask = torch.ones_like(outputs)
        mask.scatter_(1, y.unsqueeze(1), 0)
        top_k_batch = torch.topk(torch.softmax(outputs, dim=1) * mask, k=num_of_classes-1, dim=1)[1]
        for top_k_per_img in top_k_batch:
            success_class_num = 0
            for top_class in top_k_per_img:
                if len(idx_of_retain_samples_per_class[top_class])>0:
                    idx.append(idx_of_retain_samples_per_class[top_class][0])
                    idx_of_retain_samples_per_class[top_class] = idx_of_retain_samples_per_class[top_class][1:]
                    success_class_num += 1
                if success_class_num==pattern_length: break
        # breakpoint()
        top_k.append(top_k_batch)
    return np.array(idx)


def get_dataloader(args, num_of_classes, net,device):
    dataSet, transform_train, transform_valid, random_subset_idx = get_dataset_transform_randomIdx(args)
    if "cifar" in args.dataset :
        pattern_forget_data, pattern_retain_data = generate_mask_cifar(args.delta)
        transform_train = transform_train_cifar
    elif args.dataset=="tinyimagenet":
        pattern_forget_data, pattern_retain_data = generate_mask_tinyimagenet(args.delta)

    
    retainDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_idx=random_subset_idx, isForgetSet=False)
    forgetDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_idx=random_subset_idx, isForgetSet=True)
    testDS                            = dataSet("test", transform_valid)
    
    pattern_mask_for_forgetData = np.tile(pattern_forget_data, (len(forgetDS_inTrainSet_woDataAugment),1,1,1))
    pattern_mask_for_retainData = np.tile(pattern_retain_data, (len(forgetDS_inTrainSet_woDataAugment),1,1,1))
    forget_data = np.repeat(forgetDS_inTrainSet_woDataAugment.data.copy(), pattern_forget_data.shape[0], axis=0) # 沿着某一维度依次重复元素
    print(f"Mask Shape:{pattern_mask_for_forgetData.shape}\tForget Data Shape:{forget_data.shape}")
    
    forget_dl = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True)
    random_idx_of_retainData = get_random_idx(forget_dl, net, pattern_forget_data.shape[0], device, retainDS_inTrainSet_woDataAugment.label.copy(),num_of_classes)
    
    print("Success")
        
    retain_data_part = retainDS_inTrainSet_woDataAugment.data.copy()[random_idx_of_retainData].astype(np.float32)
    patch_data = retain_data_part * pattern_mask_for_retainData + forget_data.astype(np.float32) * pattern_mask_for_forgetData
    patch_data = np.clip(patch_data,0,255).astype(np.uint8)
    # save_patch_img(patch_data, pattern_forget_data.shape[0])
    random_label = retainDS_inTrainSet_woDataAugment.label.copy()[random_idx_of_retainData]
    
    forget_label_repeat = np.repeat(forgetDS_inTrainSet_woDataAugment.label, pattern_forget_data.shape[0], axis=0)
    print(f"Random Label and Forget Data Label Equal Num:{(random_label == forget_label_repeat).sum(),}\tRandomLabel:{random_label[:10]}\tForgetLabel:{forget_label_repeat[:10]}" )
    trainDS_with_patchData = ConcatDataset(
        retainDS_inTrainSet_woDataAugment.data, 
        retainDS_inTrainSet_woDataAugment.label,
        patch_data, 
        random_label,
        transform_train, 
        transform_valid)

    
    print(f"Data 1  Shape:{trainDS_with_patchData.data1.shape}\tData 2 Shape:{trainDS_with_patchData.data2.shape}")
    print("After processing, the shape of Retain Set for Training is ", len(trainDS_with_patchData))

    trainDL_with_patchData            = DataLoader(trainDS_with_patchData, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True) 
    retainDL_inTrainSet_woDataAugment = DataLoader(retainDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    forgetDL_inTrainSet_woDataAugment = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    testDL                            = DataLoader(testDS, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    

    return trainDL_with_patchData, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL, 

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
    
    trainDL_with_patchData, retainDL_inTrainSet_woDataAugment, \
        forgetDL_inTrainSet_woDataAugment, testDL = get_dataloader(args, num_of_classes=num_of_classes, net=net, device=device)
    
    
    if args.opt == 'sgd':
        optimizer       = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer       = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer       = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = None
        raise NotImplementedError
    scheduler       = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_scheduler(args, len(trainDL_with_patchData)), )
    criterion       = nn.CrossEntropyLoss()
    
    
   

    epoch_time = AverageMeter("Epoch Time")
    for epoch in range(args.epoch):
        start_time = time.time()
        log.log_print("\nEpoch:{} | Model:{} | lr now:{:.6f}".format(epoch, args.model, optimizer.state_dict()['param_groups'][0]['lr']))

        acc = train_vanilla(net, trainDL_with_patchData, scheduler, optimizer, criterion, log, device, args,)

        

        epoch_time.update(time.time() - start_time)
        print("Finished at:" + datetime.datetime.fromtimestamp(time.time() + epoch_time.val[-1]*(args.epoch -epoch)  ).strftime("Time:%H:%M"),)
    _, test_acc = valid_test(net, testDL, "Test Acc", device, criterion, log, args=args)
    _, retain_acc = valid_test(net, retainDL_inTrainSet_woDataAugment, "Retain Acc", device, criterion, log, args=args)
    _, forget_acc = valid_test(net, forgetDL_inTrainSet_woDataAugment, "Forget Acc", device, criterion, log, args=args)

    # saveModel({'net': deepcopy(net.state_dict()),}, args.ckpt.replace('.pth', '-last.pth'))
    
    net.eval()

    mia_ratio = get_membership_attack_prob(retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL, net)
    log.log_print("Test acc:{:.3f}\tRetain_acc:{:.3f}\tForget acc:{:.3f}\tMIA:{:.3f}".format(test_acc,retain_acc, forget_acc, mia_ratio, ))
    wandb.log({"test_acc": test_acc,
            "retain_acc":retain_acc,
            "forget_acc":forget_acc,
            "mia": mia_ratio})
    
    save_running_results(args)
    
    return test_acc, retain_acc, forget_acc, mia_ratio
    
def sweep_func():
    wandb.init()
    sweep_config = wandb.config
    
    args.opt = sweep_config["opt"]
    args.lr = sweep_config["lr"]
    args.wd = sweep_config["wd"]
    args.delta = sweep_config["delta"]

    repeat = 2
    avg_gap = 0
    
    for i in range(repeat):
        args.seed = i
        args.forget_per=1
        metric = main(args)
        avg_gap += get_average_gap(metric, args.dataset, args.forget_per, weight=[1,1,1,0.5])
    
    for i in range(repeat):
        args.seed = i
        args.forget_per=10
        metric = main(args)
        avg_gap += get_average_gap(metric, args.dataset, args.forget_per, weight=[1,1,1,0.5])
        
    wandb.log({"avg_gap" : avg_gap / 2 / repeat}) 
    
if __name__ == "__main__":
    args = get_args()
    args.code_file = __file__
    set_resumeCKPT(args)
    args = update_ckpt(args)
    

    
    wandb.login(key="d5322d51c6ead77187a678454f09d6cef055901c")
    if not args.debug:
        wandb.init(
            project=f"random-{args.dataset}-{args.forget_per}",
            name=f"NatMU-transfer",
            config = {"seed": args.seed, 
                      "opt": args.opt,
                      "lr":args.lr,
                      "wd":args.wd,
                      "delta":args.delta
                      },
            mode="disabled" if args.debug else "online",
        )
        main(args)
        exit()
    sweep_config = {
        'method': "bayes"
    }
    metric = {
        "name": "avg_gap",
        "goal": "minimize"
    }
    sweep_config["metric"] = metric
    
    sweep_config['parameters'] = {}
    sweep_config['parameters'].update({
    'opt': {
        'values': ['adamw']
        },
    })
    
    sweep_config['parameters'].update({
        'lr': {
            'distribution': 'uniform',
            'min': 3e-4,
            'max': 2e-3
        },
        
        'wd': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.001,
        },
        
        'delta': {
            'distribution': 'uniform',
            'min': -0.10,
            'max': 0.0,
        },
    })

    pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=f"randomsubset-sweep-NatMU-{args.dataset}")
    wandb.agent(sweep_id, sweep_func, count=100)
    # wandb.agent("hezhengbao/randomsubset-sweep-NatMU-cifar10/6z64kwve", sweep_func, count=50)
    

