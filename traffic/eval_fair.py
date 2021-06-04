
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
#import matplotlib.pyplot as plt
import PIL.Image as Image
from PIL import ImageEnhance
from tqdm import tqdm
import os
import pandas as pd
from data import data_transforms


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--model', type=str, default= 'models/fair_89.pt',
                    help='model directory')
args = parser.parse_args()


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


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/test',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=use_gpu)


def eval(model, val_loader, intensity):

    configs1 = {
        'epsilon': intensity / 255,
        'num_steps':7,
        'step_size':3/ 255,
        'clip_max': 1,
        'clip_min': 0,
        'print_process': False
    }

    all_label = []
    pred = []
    pred_adv = []

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

        all_label.append(target)
        pred.append(max_index)

        ## adv test
        adv_samples = pgd_attack(model, data, target, **configs1)
        output1 = model(adv_samples)
        max_index1 = output1.max(dim=1)[1]
        correct1 += (max_index1 == target).sum()
        pred_adv.append(max_index1)

    acc = (correct.item() / len(val_loader.dataset))
    acc1 = (correct1.item() / len(val_loader.dataset))

    all_label = torch.cat(all_label).flatten().cpu().numpy()
    pred = torch.cat(pred).flatten().cpu().numpy()
    pred_adv = torch.cat(pred_adv).flatten().cpu().numpy()

    print(acc)
    print(acc1)

    return acc, acc1, pred, pred_adv, all_label



## main function

intensity = 12
# Load network parameters
from model import Net
model = Net()
if use_gpu:
    model.cuda()
state_dict = torch.load(args.model)
model.load_state_dict(state_dict)

ds_test = test_loader

_, _, predicted, attacked, labels = eval(model, ds_test, intensity)

## finding in class accuracy and robustness

confusion_table = np.zeros([23,3])

for i in range(23):

    in_class = (labels == i)

    predicted_labels = predicted[in_class]
    attacked_labels = attacked[in_class]

    in_class_clean_acc = np.sum(predicted_labels == i)/np.sum(in_class)
    in_class_adv_acc = np.sum(attacked_labels == i)/np.sum(in_class)

    confusion_table[i, 0] = in_class_clean_acc
    confusion_table[i, 1] = in_class_adv_acc
    confusion_table[i, 2] = np.sum(in_class)

np.savetxt('results/fair_' + str(intensity) + '_.csv', confusion_table,  delimiter= ',')

