#!/bin/bash
cd ../..

seed=0
forget_per=1

python blindspot.py -r --dataset=cifar10 --model=VGG16  --epoch=5 --forget_per=$forget_per -nd --seed=$seed  --lr=0.00005 --opt="adamw" --wd=0.0009 --temp=7.2


