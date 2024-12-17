#!/bin/bash
cd ../..

seed=0
cls="rocket"

python natmu.py -r --dataset=cifar20 --ckpt=natmu-transfer.pth --epoch=5 --forget_class=$cls -nd --seed=$seed --delta=-0.053  --lr=0.0009 --opt="adamw" --wd=0.00037

