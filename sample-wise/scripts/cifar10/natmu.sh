#!/bin/bash
cd ../..

seed=0
forget_per=1

python natmu.py -r --dataset=cifar10 --model=VGG16  --epoch=5 --forget_per=$forget_per -nd --seed=$seed --delta=-0.19 --lr=0.001 --opt="adamw" --wd=0.002 

