import torch
import torch.optim as optim
from .output import AverageMeter, LogProcessBar, format_number
import time
from functools import partial
import numpy as np
import torch.nn as nn

def init_linear(net:nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))

def compute_correct(outputs: torch.tensor, targets: torch.tensor):
    _, predicted = outputs.max(1)
    return predicted.eq(targets).sum().item()


def rwp_first_step(optimizer, std=0.01,):
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.state[p]["old_p"] = p.data.clone()
            if len(p.data.shape) > 1:
                sh = p.data.shape
                sh_mul = np.prod(sh[1:])
                e_w_v = p.data.view(sh[0], -1).norm(dim=1, keepdim=True) 
                e_w = e_w_v.repeat(1, sh_mul).view(sh)
                # print (e_w.max(), e_w.min())
                e_w = torch.normal(0, (std + 1e-16) * e_w).to(p)
            else:
                e_w = torch.empty_like(p.data).to(p)
                # print (p.data.view(-1).norm().item())
                e_w.normal_(0, std * (p.data.view(-1).norm().item() + 1e-16))
            p.data.add_(e_w)  # add weight noise
            

def rwp_second_step( optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None: continue
            p.data = optimizer.state[p]["old_p"]  # get back to "w" from "w + e(w)"
            
            

def train_rwp_salun(net, trainloader, scheduler, optimizer:optim.Optimizer, criterion, log:LogProcessBar, device, args, std=0.01, mask=None):
    net.train()
    train_loss = AverageMeter("TrainLoss")
    correct_sample = AverageMeter("CorrectSample")
    total_sample = AverageMeter("TotalSample")
    training_time = AverageMeter("TrainingTime")
    
    for batch_idx, batch_data in enumerate(trainloader):
        inputs, targets = batch_data[0], batch_data[1]
        
        start_time = time.time()
        num_of_batch_samples = inputs.shape[0]
        inputs, targets = inputs.to(device), targets.to(device)

        
        optimizer.zero_grad()
        if std>0:
            rwp_first_step(optimizer, std=std)
        outputs = net(inputs)
        loss = criterion(outputs, targets.long())
        loss.backward()
        if std>0: 
            rwp_second_step(optimizer)
        if mask:
            for name, param in net.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()
        scheduler.step()
        
        train_loss.update(loss.item(), num_of_batch_samples)
        correct_sample.update(compute_correct(outputs, targets))
        total_sample.update(num_of_batch_samples)
        training_time.update(time.time() - start_time)
        
        msg = "[{}/{}] Loss:{} | Acc:{}% | {}".format(
            format_number(2, 3, training_time.avg),
            format_number(3, 2, training_time.sum),
            # datetime.datetime.fromtimestamp(time.time() + (training_time.avg*len(trainloader)+10.6)*(args.epoch -epoch)  ).strftime("Time:%H:%M"),
            format_number(1, 3, train_loss.avg),
            format_number(3, 2, 100. * correct_sample.sum / total_sample.sum),
            "Train".ljust(15),
        )
        
        if  (batch_idx == len(trainloader)-1): log.refresh(batch_idx, len(trainloader), msg)


train_vanilla = partial(train_rwp_salun, std=0, mask=None)
train_rwp = partial(train_rwp_salun, mask=None)
train_salun = partial(train_rwp_salun, std=0)
        

@torch.no_grad()
def valid_test(net, dataloader, mode: str, device, criterion, log:LogProcessBar,args, forward=None, ):
    if forward is None:
        forward = net.forward
        
    net.eval()
    train_loss = AverageMeter("TrainLoss")
    correct_sample = AverageMeter("CorrectSample")
    total_sample = AverageMeter("TotalSample")
    training_time = AverageMeter("TrainingTime")
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        start_time = time.time()
        num_of_batch_samples = inputs.shape[0]
        

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = forward(inputs)
        loss = criterion(outputs, targets)

        
        train_loss.update(loss.item(), num_of_batch_samples)
        correct_sample.update(compute_correct(outputs, targets))
        total_sample.update(num_of_batch_samples)
        training_time.update(time.time() - start_time)
        
        msg = "[{}/{}] Loss:{} | Acc:{}% | {}".format(
            format_number(2, 3, training_time.avg),
            format_number(3, 2, training_time.sum),
            format_number(1, 3, train_loss.avg),
            format_number(3, 2, 100. * correct_sample.sum / total_sample.sum),
            mode.ljust(15),
        )

        if (batch_idx == len(dataloader)-1): log.refresh(batch_idx, len(dataloader), msg)
    return train_loss.avg, 100. * correct_sample.sum / total_sample.sum

