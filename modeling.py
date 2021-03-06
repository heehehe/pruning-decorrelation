import time
import random
import pathlib
from os.path import isfile
import copy
import sys

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from resnet_mask import *
from utils import *


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device =='cuda':
        torch.cuda.manual_seed_all(777)
    
    ## args
    layers     = int(args.layers)
    prune_type = args.prune_type
    prune_rate = float(args.prune_rate)
    prune_imp  = args.prune_imp
    reg        = args.reg
    epochs     = int(args.epochs)
    batch_size = int(args.batch_size)
    lr         = float(args.lr)
    momentum   = float(args.momentum)
    wd         = float(args.wd)
    odecay     = float(args.odecay)

    if prune_type:
        prune = {'type':prune_type, 'rate':prune_rate}
    else:
        prune = None
    
    if reg == 'reg_cov':
        reg = reg_cov
    
    cfgs = {
        '18':  (BasicBlock, [2, 2, 2, 2]),
        '34':  (BasicBlock, [3, 4, 6, 3]),
        '50':  (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3]),
    }
    cfgs_cifar = {
        '20':  [3, 3, 3],
        '32':  [5, 5, 5],
        '44':  [7, 7, 7],
        '56':  [9, 9, 9],
        '110': [18, 18, 18],
    }
    
    train_data_mean = (0.5, 0.5, 0.5)
    train_data_std  = (0.5, 0.5, 0.5)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(train_data_mean, train_data_std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_data_mean, train_data_std)
    ])

    trainset    = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
    testset     = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    
    model = ResNet_CIFAR(BasicBlock, cfgs_cifar['56'], 10).to(device)
    image_size = 32
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd) #nesterov=args.nesterov)
    lr_sche   = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    ##### main 함수 보고 train 짜기
    best_acc1 = 0.0

    print('prune rate', prune_rate, 'regularization odecay', odecay)

    for epoch in range(epochs):

        acc1_train_cor, acc5_train_cor = train(trainloader, epoch=epoch, model=model, 
                                               criterion=criterion, optimizer=optimizer, 
                                               prune=prune, reg=reg, odecay=odecay)
        acc1_valid_cor, acc5_valid_cor = validate(testloader, epoch=epoch, model=model, criterion=criterion)

        acc1_train = round(acc1_train_cor.item(), 4)
        acc5_train = round(acc5_train_cor.item(), 4)
        acc1_valid = round(acc1_valid_cor.item(), 4)
        acc5_valid = round(acc5_valid_cor.item(), 4)

        # remember best Acc@1 and save checkpoint and summary csv file
    #     summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]

        is_best = acc1_valid > best_acc1
        best_acc1 = max(acc1_valid, best_acc1)
        if is_best:
            summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]
    print(summary)
    #     save_model(arch_name, args.dataset, state, args.save)
    #     save_summary(arch_name, args.dataset, args.save.split('.pth')[0], summary)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--layers', default=56)
    parser.add_argument('--prune_type', default=None, help='None / structured / unstructured')
    parser.add_argument('--prune_rate', default=0.9)
    parser.add_argument('--prune_imp', default='L2')
    parser.add_argument('--reg', default=None, help='None / reg_cov')
    parser.add_argument('--epochs', default=300)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--lr', default=0.2)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--wd', default=1e-4)
    parser.add_argument('--odecay', default=1)
    args = parser.parse_args()
    
    main(args)
