#!/bin/bash
cd ../..

seed=0
cls="rocket"

python salun.py -r --dataset=cifar20 --epoch=5 --forget_class=$cls -nd --seed=$seed  \
        --lr=0.00017 --opt=adamw --wd=0 --threshold=0.01 --ckpt=salun-transfer.pth
