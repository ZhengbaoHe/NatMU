#!/bin/bash
seed=0
cls="rocket"

python salun.py -r --dataset=cifar100 --epoch=5 --forget_class=$cls -nd --seed=$seed  \
    --lr=0.00013 --opt=adamw --wd=0 --threshold=0.7 --ckpt=salun-transfer.pth

