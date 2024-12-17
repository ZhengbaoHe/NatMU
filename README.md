# Towards Natural Machine Unlearning

This is the official code repository for paper: [Towards Natural Machine Unlearning](https://arxiv.org/abs/2405.15495).

---
## Instructions for the Code of NatMU

Our code's directory structure is as following:
```
.
|-- data
|-- class_wise
`-- sample_wise
```
``data`` folder includes the nessenary dataset, and you should download them later.
``class-wise`` folder includes the code for full-class and sub-class unlearning.
``sample-wise`` folder includes the code for sample-wise unlearning.


### 1. Prepare Environment
Our code can be run in a common pytorch environment. The experiments are done using the following packages:
```
matplotlib==3.4.1
numpy==1.20.1
Pillow==9.2.0
Pillow==11.0.0
scikit_learn==0.24.2
torch==1.12.0+cu113
torchtoolbox==0.1.8.2
torchvision==0.9.0a0
wandb==0.15.12
```

### 2. Prepare Data

Download the  ``data`` folder for experiments from [Google Drive](https://drive.google.com/drive/folders/1n5nY_Q2e7mqclA3aA6AkwWqt4zR63dnP?usp=sharing) to prepare nessenary data. The resulting directory structure tree is like:
```
data/
|-- cifar10
|   |-- batches.meta
|   |-- data_batch_1
|   |-- data_batch_2
|   |-- data_batch_3
|   |-- data_batch_4
|   |-- data_batch_5
|   |-- readme.html
|   `-- test_batch
|-- cifar100
|   |-- file.txt~
|   |-- meta
|   |-- test
|   `-- train
`-- tiny-imagenet-200
    |-- train_data.npy
    |-- train_label.npy
    |-- val_data.npy
    `-- val_label.npy
```


### 3. Get the Unlearned Model

We take the sample-wise unlearning as an example. You can conduct the same procedure for class-wise unlearning. 


Enter the sample-wise folder:
```bash
cd sample-wise
```


#### 3.1 Training the Pretrain Model

Firstly, we should obtain the pretrain model with the following script:
```bash
python train_origin.py --model=VGG16 --dataset=cifar10 --opt=sgd --lr=0.1 --lr_scheduler=step --wd=5e-4 --batchsize=128 -nd

python train_origin.py --model=ResNet18 --dataset=cifar100 --opt=sgd --lr=0.1 --lr_scheduler=step --wd=5e-4 --batchsize=128 -nd

python train_origin.py --model=ResNet34 --dataset=tinyimagenet --opt=sgd --lr=0.1 --lr_scheduler=step --wd=5e-4 --batchsize=256 -nd
```

The resulting checkpoint is saved in folder ``ckpt/<model_name>/Vanilla``. Then, you should set the ``resumeCKPT`` argument in the function ``set_resumeCKPT(): line34 line36 line42`` in ``utils/unlearning_util.py``.

We also provide pretrain checkpoints in [Google Drive](https://drive.google.com/drive/folders/1n5nY_Q2e7mqclA3aA6AkwWqt4zR63dnP?usp=sharing). You can also download the checkpoints.



#### 3.2 Running Unearning Methods

In the ``scripts`` folder, we provide the running script for different methods with pre-searched hyperparameters. You can easily get the unlearned model by running the scripts. For example:

> The ``idx`` folder saves the sample index to be forgotten. Set the argument ``forget_per`` in code to forget differnet ratios of forgetting samples.
> ``forget_per`` = 1 /10 means forgetting 1%/10 % samples which are randomly selected.
> ``forget_per`` = 101 /110 means forgetting 1%/10 % samples which are the most difficult to learn.

```bash
cd scripts/cifar10
bash natmu.sh
```
The log file would be saved at ``ckpt/<model_name>/<method>``. The dataset information and forgeting ratio is coded in the  children path.

- To forget to most-difficult-to-learn samples, you should run ``natmu_difficult_sample.py`` to avoid that there are not enough remaining samples of a particular class to generate unlearning instances.
- In some codes, the method BadTeacher is also named as "blindspot"

#### 3.3 Class-wise Unlearning

In class-wise unlearning, we use an argument ``forget_class`` to choose different forgetting calsses. Different to sample-wise unlearning, the resulting checkpoint is saved in folder ``ckpt/<model_name>/<dataset>/<method>``. 