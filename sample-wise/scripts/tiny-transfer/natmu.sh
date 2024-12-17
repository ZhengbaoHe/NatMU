#!/bin/bash
cd ../..

seed=0
forget_per=1

python natmu.py -r --dataset=tinyimagenet --model=ResNet34 --batchsize=256 --epoch=5 --forget_per=$forget_per -nd --seed=$seed --delta=0.045 --lr=0.00083 --opt="adamw" --wd=0.00007

