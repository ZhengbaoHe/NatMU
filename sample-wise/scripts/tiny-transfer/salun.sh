#!/bin/bash
cd ../..



seed=0
forget_per=1
        
python salun.py -r --dataset=tinyimagenet --model=ResNet34 --batchsize=256 --epoch=5 --forget_per=$forget_per -nd --seed=$seed  --lr=0.000055 --opt="adamw" --wd=0.00053 --threshold=0.20
