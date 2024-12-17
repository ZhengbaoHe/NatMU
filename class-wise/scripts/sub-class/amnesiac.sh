#!/bin/bash
cd ../..

seed=0
cls="rocket"

python amnesiac.py -r --dataset=cifar20 --epoch=5 --forget_class=$cls -nd --seed=$seed  --lr=0.000018 --opt=adamw --wd=0.00037 -i --ckpt=amnesiac-transfer.pth
