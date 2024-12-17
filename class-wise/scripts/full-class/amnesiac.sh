#!/bin/bash
cd ../..

seed=0
cls="rocket"

python amnesiac.py -r --dataset=cifar100 --epoch=5 --forget_class=$cls -nd --seed=$seed  --lr=0.000145 --opt=adamw --wd=0.00008 --ckpt=amnesiac-transfer.pth
