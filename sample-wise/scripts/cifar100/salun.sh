#!/bin/bash
cd ../..

seed=0
forget_per=1

python salun.py -r --dataset=cifar100 --model=ResNet18  --epoch=5 --forget_per=$forget_per -nd --seed=$seed  --lr=0.000057 --opt="adamw" --wd=0.001 --threshold=0.21
