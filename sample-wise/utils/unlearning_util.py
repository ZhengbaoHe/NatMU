from .data import *

def get_dataset_transform_randomIdx(args):
    if args.dataset=="cifar10" :
        dataSet = DatasetCifar10
        transform_train = transform_train_cifar
        transform_valid = transform_valid_cifar
        height_and_width = 32
    elif args.dataset=="cifar100":
        dataSet = DatasetCifar100
        transform_train = transform_train_cifar
        transform_valid = transform_valid_cifar
        height_and_width = 32
    elif args.dataset=="tinyimagenet":
        dataSet = DatasetTinyImageNet
        transform_train = transform_train_tinyimagenet
        transform_valid = transform_valid_tinyimagenet
        height_and_width = 64
    else:
        raise NotImplementedError(f"Not implement this dataset : {args.dataset}")
    if hasattr(args, "forget_per"):
        random_subset_idx = get_random_subset_idx(args.dataset, args.forget_per)
        print("Random subset Length:", len(random_subset_idx))
        return dataSet, transform_train, transform_valid, random_subset_idx 
    return dataSet, transform_train, transform_valid

def set_resumeCKPT(args):
    if "retrain" in args.code_file or "finetune_from_init" in args.ckpt:
        mode = 'init'
    else:
        mode = 'last'
    if args.resumeCKPT is None and "ResNet" in args.model:
        if args.dataset == "cifar100":
            args.resumeCKPT = f"ckpt/ResNet18/Vanilla/24_10_08_22_05_52.65_cifar100_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-{mode}.pth"
        elif args.dataset == "tinyimagenet":
            args.resumeCKPT = f"ckpt/ResNet34/Vanilla/24_10_08_22_23_17.02_tinyimagenet_sgd_lr-0.1_epoch-100_batchsize-256_wd-0.0005/model-{mode}.pth"
        else:
            raise NotImplementedError
        return
    if args.resumeCKPT is None and  args.model == "VGG16":
        if args.dataset == "cifar10":
            args.resumeCKPT = f"ckpt/VGG16/Vanilla/24_10_08_21_51_04.29_cifar10_sgd_lr-0.1_epoch-100_batchsize-128_wd-0.0005/model-{mode}.pth"
        else:
            raise NotImplementedError
    
        
        
        
        
from copy import deepcopy
def get_retrain_models(net, args, device):
    retrain_models = []
    root = f"ckpt/{args.model}/retrain_steplr"
    checkpoints = os.listdir(root)
    for checkpoint in checkpoints:
        if f"_{args.dataset}_" not in checkpoint:
            continue
        retrain_model = deepcopy(net)
        retrain_model.load_state_dict(torch.load(os.path.join(root, checkpoint, "model-last.pth"))["net"])
        retrain_models.append(retrain_model.to(device))
    return retrain_models
