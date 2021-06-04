
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
from data import initialize_data, data_transforms


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--model', type=str, default= 'model_5.pth',
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


# Load network parameters
from model import Net
model = Net()
if use_gpu:
    model.cuda()
state_dict = torch.load(args.model)
model.load_state_dict(state_dict)



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')



def eval():
    configs1 = {
        'epsilon': 12 / 255,
        'num_steps': 15,
        'step_size': 1 / 255,
        'clip_max': 1,
        'clip_min': 0,
        'print_process': False
    }

    df = pd.read_csv('data/final_test.csv', sep=';')
    names = df['Filename']
    labels = df['ClassId']
    labels = np.array(labels, dtype= int).flatten()
    correct = 0
    correct1 = 0
    all = 0
    all_adv = 0

    model.eval()
    test_dir = args.data + '/test_images'

    for f in tqdm(os.listdir(test_dir)):
        if 'ppm' in f:

            ## clean acc
            output = torch.zeros([1, 43], dtype=torch.float32, device = 'cuda')
            with torch.no_grad():
                data = data_transforms(pil_loader(test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                data = Variable(data).cuda()
                output = output.add(model(data))
            pred = torch.argmax(output)
            id = np.where(names == f)[0]
            correct = correct + (labels[id].item() == pred.item())
            all = all + 1

            ## adv acc
            pred = pred.unsqueeze(0)
            output1 = torch.zeros([1, 43], dtype=torch.float32, device = 'cuda')
            if labels[id].item() == pred.item():
                adv_samples = pgd_attack(model, data, pred, **configs1)
                output1 = output1.add(model(adv_samples))
                pred1 = torch.argmax(output1)
                correct1 = correct1 + (pred.item() == pred1.item())
                all_adv = all_adv + 1

    acc = correct/all
    adv_acc = correct1/all_adv

    print('test acc = ' + str(correct/all))
    print('adv acc = ' + str(correct1/all_adv))

    return acc, adv_acc


## main function
eval()
