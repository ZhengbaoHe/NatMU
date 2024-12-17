#!/bin/bash
cd ../..

seed=0
forget_per=1

python natmu.py -r --dataset=cifar100 --model=ResNet18  --epoch=5 --forget_per=$forget_per -nd --seed=$seed --delta=-0.03 --lr=0.0011 --opt="adamw" --wd=0.0005 

