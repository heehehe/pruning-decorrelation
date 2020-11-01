import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import masknn
import resnet_mask

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

import sys


def get_weight_threshold(model, rate, prune_imp='L1'):
    importance_all = None
    for name, item in model.named_parameters():  
        #module.named_parameters():
        if len(item.size())==4 and 'mask' not in name:
            weights = item.data.view(-1).cpu()
            grads = item.grad.data.view(-1).cpu()

            if prune_imp == 'L1':
                importance = weights.abs().numpy()
            elif prune_imp == 'L2':
                importance = weights.pow(2).numpy()
            elif prune_imp == 'grad':
                importance = grads.abs().numpy()
            elif prune_imp == 'syn':
                importance = (weights * grads).abs().numpy()
            

            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)

    threshold = np.sort(importance_all)[int(len(importance_all) * rate)]
    return threshold

# %% [code]
def weight_prune(model, threshold, prune_imp='L1'):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'weight' in name:
            key = name.replace('weight', 'mask')
            if key in state.keys():
                if prune_imp == 'L1':
                    mat = item.data.abs()
                elif prune_imp == 'L2':
                    mat = item.data.pow(2)
                elif prune_imp == 'grad':
                    mat = item.grad.data.abs()
                elif prune_imp == 'syn':
                    mat = (item.data * item.grad.data).abs()
                state[key].data.copy_(torch.gt(mat, threshold).float())

# %% [code]
def get_filter_mask(model, rate, prune_imp='L1'):
    importance_all = None
    for name, item in model.named_parameters():  
        #.module.named_parameters():
        if len(item.size())==4 and 'weight' in name:
            filters = item.data.view(item.size(0), -1).cpu()
            weight_len = filters.size(1)
            if prune_imp =='L1':
                importance = filters.abs().sum(dim=1).numpy() / weight_len
            elif prune_imp == 'L2':
                importance = filters.pow(2).sum(dim=1).numpy() / weight_len
        
            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)
                

    threshold = np.sort(importance)[int(len(importance) * rate)]
    #threshold = np.percentile(importance, rate)
    filter_mask = np.greater(importance, threshold)
    return filter_mask

# %% [code]
def filter_prune(model, filter_mask):
    idx = 0
    for name, item in model.named_parameters():  
        #.module.named_parameters():
        if len(item.size())==4 and 'mask' in name:
            for i in range(item.size(0)):
                item.data[i,:,:,:] = 1 if filter_mask[idx] else 0
                idx += 1

# %% [code]
def cal_sparsity(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.named_parameters():  
        #.module.named_parameters():
        if 'mask' in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        if 'weight' in name or 'bias' in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity



def reg_ortho(mdl):
    l2_reg = None
    for W in mdl.parameters():
        if W.ndimension() < 2:
            continue
        else:
            cols = W[0].numel()
            rows = W.shape[0]
            w1 = W.view(-1,cols)
            wt = torch.transpose(w1,0,1)
            m  = torch.matmul(wt,w1)
            ident = Variable(torch.eye(cols,cols))
            ident = ident.cuda()

            w_tmp = (m - ident)
            height = w_tmp.size(0)
            u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            if l2_reg is None:
                l2_reg = (sigma)**2
            else:
                l2_reg = l2_reg + (sigma)**2
    return l2_reg


def reg_cov(mdl):
    cov_reg = 0
    for W in mdl.parameters():
        if W.ndimension() < 2:
            continue
        else:
            for w in W:
                for w_ in w:
                    if w_.dim() > 0 and len(w_) == 2:
                        cov_ = np.cov(w_.detach().numpy())
                        cov_upper = np.triu(cov_)
                        cov_upper_abs = np.absolute(cov_upper)
                        cov_upper_abs_sum = np.sum(cov_upper_abs)
                        cov_reg += cov_upper_abs_sum
            
    return cov_reg


class AverageMeter(object):
    r"""Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

def accuracy(output, target, topk=(1,)):
    r"""Computes the accuracy over the $k$ top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, epoch, model, criterion, optimizer, reg=None, prune=None, odecay=0, device='cuda'):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.train()
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        try:
            if prune:
                if prune['type'] == 'structured':
                    filter_mask = get_filter_mask(model, prune['rate'])
                    filter_prune(model, filter_mask)
                elif prune['type'] == 'unstructured':
                    thres = get_weight_threshold(model, prune['target_sparsity'])
                    weight_prune(model, thres)        
            outputs = model(inputs)
            if prune['type'] == 'structured': print('PRUNED in', i, 'with length of', len(filter_mask))
            elif prune['type'] == 'unstructured': print('PRUNED in', i, 'with thres', thres)
        except:
            outputs = model(inputs)
            
        if reg:
            oloss = reg(model)
            oloss = odecay * oloss
            loss = criterion(outputs, targets) + oloss
        else:
            loss = criterion(outputs, targets)
        
        acc1, acc5 = accuracy(outputs, targets, topk=(1,5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('train {i} ====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(i=epoch, top1=top1, top5=top5))
    return top1.avg, top5.avg


def validate(val_loader, epoch, model, criterion, device='cuda'):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            acc1, acc5 = accuracy(outputs, targets, topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            
    print('valid {i} ====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(i=epoch, top1=top1, top5=top5))
    return top1.avg, top5.avg