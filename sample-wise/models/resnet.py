'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import os
import numpy as np, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

momentum = 0.1  # default 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(1, 1), stride=(stride, stride), bias=False),
                nn.BatchNorm2d(self.expansion * planes, momentum=momentum)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, momentum=momentum)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(1,1), stride=(stride), bias=False),
                nn.BatchNorm2d(self.expansion * planes, momentum=momentum)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=momentum)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.layer4[-1].register_forward_hook(self._get_features_hook)
        self.layer4[-1].register_full_backward_hook(self._get_grads_hook)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        self.idx = []


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _get_features_hook(self, module, input, output):
        self.feature = output.detach()
        
    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        self.gradient = output_grad[0].detach()
        
    

    def forward(self, x):
        def hook(grad):
            self.grad = grad.detach()

        out = F.relu(self.bn1(self.conv1(x)))
        for i in self.idx: out[:,i,:,:] = 0
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_features(self, x):
        out = self.conv1(x)
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = F.adaptive_avg_pool2d(out, (1,1))
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out
        
    
    def forward_(self, x):
        f0 = F.relu(self.bn1(self.conv1(x)))

        with torch.no_grad():
            mean = torch.mean(f0.detach(), dim=(0, 2, 3))
            var = torch.var(f0.detach(), dim=(0, 2, 3))
            n = torch.numel(f0.detach()) / f0.shape[0]
            if self.n == 0 :
                self.mean = mean
                self.var = var 
                self.n = n
            else:
                sum_n = (self.n + n)
                self.var = (n * var + self.n * self.var + self.n * n * torch.pow(mean - self.mean, exponent=2) / sum_n) / sum_n
                self.mean = (mean * n + self.mean * self.n) / sum_n
                self.n = sum_n


        out = self.layer1(f0)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # self.grad = out.clone()
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    

    def forward_masked(self, x):
        f0 = F.relu(self.bn1(self.conv1(x)))

        with torch.no_grad():
            mean = torch.mean(f0.detach(), dim=(0, 2, 3))
            var = torch.var(f0.detach(), dim=(0, 2, 3))
            n = torch.numel(f0.detach()) / f0.shape[0]
            if self.n == 0 :
                self.mean = mean
                self.var = var 
                self.n = n
            else:
                sum_n = (self.n + n)
                self.var = (n * var + self.n * self.var + self.n * n * torch.pow(mean - self.mean, exponent=2) / sum_n) / sum_n
                self.mean = (mean * n + self.mean * self.n) / sum_n
                self.n = sum_n
        mask = torch.ones((1,self.var.numel(),1,1,)).to(f0)
        for i in torch.argsort(self.var[-7:]):
            mask[:,i,:,:] = 0


        out = self.layer1(f0 * mask.detach())
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        # self.grad = out.clone()
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out






        





def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)




def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)



