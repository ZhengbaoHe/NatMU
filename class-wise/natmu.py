import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
from utils import *
import time, datetime, os
from pprint import pprint

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
    parser.add_argument('--dataset', type=str, default='cifar100')
    
    parser.add_argument('--forget_class', type=str)
    parser.add_argument('--init_linear', "-i", action='store_true', help="init the linear layer weights before unleanring, this is helpful in class-wise unleanring for some methods")
    
    parser.add_argument('--delta', type=float, default=1.)
    

    args = parser.parse_args()
    return args


@torch.no_grad()
def select_patch_class(forget_loader, net, pattern_length, device):
    net.eval()
    top_k = []
    for x,y in forget_loader:
        x, y = x.to(device), y.to(device)
        outputs = net(x)
        mask = torch.ones_like(outputs)
        mask.scatter_(1, y.unsqueeze(1), 0)
        top_k_batch = torch.topk(torch.softmax(outputs, dim=1) * mask, k=pattern_length, dim=1)[1]
        top_k.append(top_k_batch)
    top_k = torch.cat(top_k, dim=0)
    return top_k.reshape(-1)


def get_random_idx_of_retainData_from_randomLabel(retain_labels, random_label,):
    random_idx_of_retainset = np.random.permutation(len(retain_labels))
    length = len(random_label)
    position_eq = retain_labels[random_idx_of_retainset[:length]] == random_label
    while position_eq.sum()<length:
        idx_eq = np.nonzero(position_eq)[0]
        eq_mask = np.isin(np.arange(len(random_idx_of_retainset)), idx_eq)
        random_idx_of_retainset[~eq_mask] = np.random.permutation(random_idx_of_retainset[~eq_mask])
        position_eq = retain_labels[random_idx_of_retainset[:length]] == random_label
    return random_idx_of_retainset[:length]

def get_dataloader(args, num_of_classes, net, device):
    
    if "cifar" in args.dataset:
        pattern_forget_data, pattern_retain_data = generate_mask_cifar(args.delta) # shape (4,3,32,32)
        height_and_width = 32
    
    dataSet, transform_train, transform_valid, forget_class_int = get_dataset_transforms(args)
    
    
    retainDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_class=forget_class_int, isForgetSet=False)
    forgetDS_inTrainSet_woDataAugment = dataSet("train", transform=transform_valid, forget_class=forget_class_int, isForgetSet=True)
    retainDS_inTestSet                = dataSet("test", transform=transform_valid, forget_class=forget_class_int, isForgetSet=False)
    forgetDS_inTestSet                = dataSet("test", transform=transform_valid, forget_class=forget_class_int, isForgetSet=True)
    testDS                            = dataSet("test", transform=transform_valid)
    
    pattern_mask_for_forgetData = np.tile(pattern_forget_data, (len(forgetDS_inTrainSet_woDataAugment),1,1,1)) # (4,3,32,32) -> (4*num_forget_data, 3, 32, 32)
    pattern_mask_for_retainData = np.tile(pattern_retain_data, (len(forgetDS_inTrainSet_woDataAugment),1,1,1))
    forget_data = np.repeat(forgetDS_inTrainSet_woDataAugment.data.copy(), pattern_forget_data.shape[0], axis=0).astype(np.float32) # 沿着某一维度依次重复元素

    print(f"Mask Shape:{pattern_mask_for_forgetData.shape}\tForget Data Shape:{forget_data.shape}")
    forget_dl = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True)
    random_label = select_patch_class(forget_dl , net, pattern_forget_data.shape[0], device).cpu().numpy()
    forget_label_repeat = np.repeat(forgetDS_inTrainSet_woDataAugment.label, pattern_forget_data.shape[0], axis=0)
    
    random_idx_of_retainData = get_random_idx_of_retainData_from_randomLabel(retainDS_inTrainSet_woDataAugment.label.copy(), random_label,)
    
    # random_idx_of_retainData = get_pure_random_idx_of_retainData(
    #                             retain_labels=retainDS_inTrainSet_woDataAugment.label.copy(), 
    #                             forget_labels=forgetDS_inTrainSet_woDataAugment.label.copy(), 
    #                             pattern_length=pattern_forget_data.shape[0], 
    #                             num_of_classes=num_of_classes)
    
    
    retain_data_part = retainDS_inTrainSet_woDataAugment.data.copy()[random_idx_of_retainData].astype(np.float32) * pattern_mask_for_retainData
    patch_data = retain_data_part  + forget_data * pattern_mask_for_forgetData
    patch_data = np.clip(patch_data,0,255).astype(np.uint8)
    # save_patch_img(patch_data, pattern_forget_data.shape[0])
    random_label = retainDS_inTrainSet_woDataAugment.label.copy()[random_idx_of_retainData]

    retain_data, retain_label = retainDS_inTrainSet_woDataAugment.data, retainDS_inTrainSet_woDataAugment.label
    
    
    print(f"Random Label and Forget Data Label Equal Num:{(random_label == forget_label_repeat).sum(),}")
    print(f"RandomLabel:{random_label[:15]}\tForgetLabel:{forget_label_repeat[:15]}")
    trainDS_with_patchData = ConcatDataset(
        retain_data, 
        retain_label,
        patch_data,
        random_label,
        transform_train, 
        transform_valid)

    
    print(f"Data 1  Shape:{trainDS_with_patchData.data1.shape}\tData 2 Shape:{trainDS_with_patchData.data2.shape}")
    print("After processing, the shape of Retain Set for Training is ", len(trainDS_with_patchData))

    trainDL_with_patchData            = DataLoader(trainDS_with_patchData, batch_size=args.batchsize, shuffle=True, num_workers=8, pin_memory=True) 
    retainDL_inTrainSet_woDataAugment = DataLoader(retainDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    forgetDL_inTrainSet_woDataAugment = DataLoader(forgetDS_inTrainSet_woDataAugment, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    retainDL_inTestSet                = DataLoader(retainDS_inTestSet, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    forgetDL_inTestSet                = DataLoader(forgetDS_inTestSet, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 
    testDL                            = DataLoader(testDS, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True) 


    return trainDL_with_patchData, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, retainDL_inTestSet, forgetDL_inTestSet, testDL, 


def main(args=None):
    if args is None:
        args = get_args()
        args.code_file = __file__
        set_resumeCKPT(args)
        args = update_ckpt(args)
    pprint(args)
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
    tmp_dataloaders = get_dataloader(args, num_of_classes, net, device)
    for dataloader in tmp_dataloaders:
        log.log_print(get_dataset_info(dataloader))
        
    trainDL_with_patchData, retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, \
        retainDL_inTestSet, forgetDL_inTestSet, testDL = tmp_dataloaders
    
    
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
    

    
    if args.init_linear: init_linear(net)
    epoch_time = AverageMeter("Epoch Time")
    for epoch in range(args.epoch):
        start_time = time.time()
        log.log_print("\nEpoch:{} | Model:{} | lr now:{:.6f}".format(epoch, args.model, optimizer.state_dict()['param_groups'][0]['lr']))

        train_vanilla(net, trainDL_with_patchData, scheduler, optimizer, criterion, log, device, args)
        _, test_acc = valid_test(net, retainDL_inTestSet, "Test Acc on Test", device, criterion, log, args=args)
        _, retain_acc = valid_test(net, retainDL_inTrainSet_woDataAugment, "Retain Acc on Test", device, criterion, log, args=args)
        _, forget_acc_on_train = valid_test(net, forgetDL_inTrainSet_woDataAugment, "Forget Acc on Train", device, criterion, log, args=args)
        _, forget_acc_on_test = valid_test(net, forgetDL_inTestSet, "Forget Acc on Test", device, criterion, log, args=args)
        
        epoch_time.update(time.time() - start_time)
        print("Finished at:" + datetime.datetime.fromtimestamp(time.time() + epoch_time.val[-1]*(args.epoch -epoch)  ).strftime("Time:%H:%M"),)
        

    
    mia_ratio = get_membership_attack_prob(retainDL_inTrainSet_woDataAugment, forgetDL_inTrainSet_woDataAugment, testDL, net)
    log.log_print(f"Test Acc:{test_acc:.3f}\tRetain Acc:{retain_acc:.3f}\tForget Acc on Train:{forget_acc_on_train:.3f}\tForget Acc on Test:{forget_acc_on_test:.3f}\tMIA:{mia_ratio:.3f}")

    
    save_running_results(args)
    
    return test_acc, retain_acc, forget_acc_on_train, forget_acc_on_test, mia_ratio



    
    
    
if __name__ == "__main__":
    args = get_args()
    args.code_file = __file__
    set_resumeCKPT(args)
    args = update_ckpt(args)
    
    
    main(args)
    