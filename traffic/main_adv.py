
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm
import os
import pandas as pd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data/', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
torch.manual_seed(args.seed)


if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
    use_gpu = False
    print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor


### Data Initialization and Loading
from data import data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation, \
    data_jitter_contrast, data_rotate, data_hvflip, data_shear, data_translate, data_center, data_hflip, \
    data_vflip  # data.py in the same folder

# Apply data transformations on the training images to augment dataset
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([datasets.ImageFolder(args.data + 'train/',
                                                         transform=data_transforms),
                                    datasets.ImageFolder(args.data+ 'train/',
                                                         transform=data_jitter_brightness),
                                    datasets.ImageFolder(args.data+ 'train/',
                                                         transform=data_jitter_hue),
                                    datasets.ImageFolder(args.data+ 'train/',
                                                         transform=data_jitter_contrast),
                                    datasets.ImageFolder(args.data+ 'train/',
                                                         transform=data_jitter_saturation),
                                    datasets.ImageFolder(args.data+ 'train/',
                                                         transform=data_translate),
                                    datasets.ImageFolder(args.data+ 'train/',
                                                         transform=data_rotate),
                                    #datasets.ImageFolder(args.data + '/train_images',
                                    #                     transform=data_hvflip),
                                    datasets.ImageFolder(args.data+ 'train/',
                                                         transform=data_center),
                                    datasets.ImageFolder(args.data+ 'train/',
                                                         transform=data_shear)]), batch_size=args.batch_size,
    shuffle=True, num_workers=1, pin_memory=use_gpu)


val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/validation',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=use_gpu)


def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size,
                  print_process):

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    #TODO: find a other way
    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)

        if print_process:
            print("iteration {:.0f}, loss:{:.4f}".format(i,loss))

        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd


def train(epoch, model, train_loader):

    configs = {
        'epsilon': 16 / 255,
        'num_steps': 7,
        'step_size': 4 / 255,
        'clip_max': 1,
        'clip_min': 0,
        'print_process': False
    }

    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)

        if use_gpu:
            data = data.cuda()
            target = target.cuda()


        ## generate PGD attack
        adv_samples = pgd_attack(model, data, target, **configs)

        optimizer.zero_grad()
        output = model(adv_samples)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        max_index = output.max(dim=1)[1]
        correct += (max_index == target).sum()
        training_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item() / (args.batch_size * args.log_interval),
                loss.data.item()))

    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def eval(model, val_loader):


    configs1 = {
        'epsilon': 16 / 255,
        'num_steps': 7,
        'step_size': 4 / 255,
        'clip_max': 1,
        'clip_min': 0,
        'print_process': False
    }


    model.eval()
    correct = 0
    correct1 = 0

    for batch_idx, (data, target) in enumerate(val_loader):
        data, target = Variable(data), Variable(target)

        if use_gpu:
            data = data.cuda()
            target = target.cuda()

        ## clean test
        output = model(data)
        max_index = output.max(dim=1)[1]
        correct += (max_index == target).sum()

        ## adv test
        adv_samples = pgd_attack(model, data, target, **configs1)
        output1 = model(adv_samples)
        max_index1 = output1.max(dim=1)[1]
        correct1 += (max_index1 == target).sum()

    acc = (correct.item() / len(val_loader.dataset))
    acc1 = (correct1.item() / len(val_loader.dataset))

    return acc, acc1


################# main function
# Neural Network and Optimizer
from model import Net
model = Net().cuda()

## optimizers
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[13, 20, 30], gamma=0.2)

## data loaders
ds_train = train_loader
ds_val = val_loader

all_acc = []
for epoch in range(1, args.epochs + 1):
    train(epoch, model, ds_train)
    lr_scheduler.step()
    acc, acc_adv = eval(model, ds_val)

    model_file = 'models/badv_model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file)

    all_acc.append(acc_adv)
    result = np.array(all_acc)
    print(result)
    if epoch > 1:
        id = np.argmax(result)
        print(result)
        print('best epoch' + str(id + 1))
