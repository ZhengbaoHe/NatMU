#!/bin/bash
cd ../..

seed=0
forget_per=1

python blindspot.py -r --dataset=cifar100 --model=ResNet18  --epoch=5 --forget_per=$forget_per -nd --seed=$seed  --lr=0.00028 --opt="adamw" --wd=0.0009 --temp=2.4
