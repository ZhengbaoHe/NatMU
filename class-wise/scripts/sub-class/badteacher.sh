#!/bin/bash
cd ../..

seed=0
cls="rocket"

python blindspot.py -r --dataset=cifar20 --epoch=5 --forget_class=$cls -nd --seed=$seed  \
         --lr=0.000074 --opt="adamw" --temp=0.72 --wd=0.0 --ckpt=blindspot-transfer.pth
