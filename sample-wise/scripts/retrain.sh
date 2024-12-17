#!/bin/bash


cd ..

seed=0
forget_per=1

python retrain.py --model=VGG16 --dataset=cifar10 --opt=sgd --lr=0.1 --lr_scheduler=step --wd=5e-4 --batchsize=128 -nd --forget_per=$per  --seed=$seed -r

python retrain.py --model=ResNet18 --dataset=cifar100 --opt=sgd --lr=0.1 --lr_scheduler=step --wd=5e-4 --batchsize=128 -nd --forget_per=$per --seed=$seed -r 

python retrain.py --model=ResNet34 --dataset=tinyimagenet --opt=sgd --lr=0.1 --lr_scheduler=step --wd=5e-4 --batchsize=256 -nd --forget_per=$per --seed=$seed -r


