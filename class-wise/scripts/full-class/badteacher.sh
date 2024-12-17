#!/bin/bash
cd ../..

seed=0
cls="rocket"

python blindspot.py -r --dataset=cifar100 --epoch=5 --forget_class=$cls -nd --seed=$seed  \
-i --lr=0.000034 --opt="adamw" --temp=0.77 --wd=0 --ckpt=blindspot-transfer.pth
