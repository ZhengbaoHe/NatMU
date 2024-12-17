#!/bin/bash
cd ../..

seed=0
forget_per=1

python salun.py -r --dataset=cifar10 --model=VGG16  --epoch=5 --forget_per=$forget_per -nd --seed=$seed  --lr=0.00005 --opt="adamw" --wd=0.0003 --threshold=0.05

