#!/bin/bash
cd ../..

seed=0
cls="rocket"

python natmu.py -r --dataset=cifar100 --epoch=5 --forget_class=$cls -nd --seed=$seed --delta=-0.053 -i --lr=0.00028 --opt="adamw" --wd=0.0 --ckpt=natmu-transfer.pth
