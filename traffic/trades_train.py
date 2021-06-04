from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from deeprobust.image.attack.pgd import PGD



def assign_model(model, device = 'cuda'):

    if (model == 'ResNet18'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet18().to(device)

    elif (model == 'ResNet34'):
        import deeprobust.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet34().to(device)

    elif (model == 'VGG16'):
        import deeprobust.image.netmodels.vgg as MODEL
        train_net = MODEL.VGG('VGG16').to(device)

    elif (model == 'DenseNet121'):
        import deeprobust.image.netmodels.densenet as MODEL
        train_net = MODEL.DenseNet121().to(device)

    return train_net




def feed_dataset(data, data_dict):
    if(data == 'CIFAR10'):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        train_loader = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=True, download = True,
                        transform=transform_train),
                 batch_size= 100, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=False, download = True,
                        transform=transform_val),
                batch_size= 100, shuffle=True) #, **kwargs)


    return train_loader, test_loader




def trades_adv(model,
               x_natural,
               step_size= 2 / 255,
               epsilon= 8 /255,
               perturb_steps= 10,
               distance='l_inf'):

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average = False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv




def train(model, train_loader, optimizer, epoch, beta, device):

    criterion_kl = nn.KLDivLoss(size_average = False)
    print('now epoch:  ' + str(epoch))
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)

        logits = model(data)
        loss_natural = F.cross_entropy(logits, target)

        x_adv = trades_adv(model, data)
        batch_size = data.shape[0]

        optimizer.zero_grad()
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                        F.softmax(model(data), dim=1))
        loss = loss_natural + beta * loss_robust
        loss.backward()
        optimizer.step()


def test(model, test_loader, adversary, configs1, device):

    print('Doing test')
    model.eval()

    test_loss = 0
    correct = 0

    test_loss_adv = 0
    correct_adv = 0

    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)

        ## clean test
        output = model(data)
        #test_loss += F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        model.zero_grad()


        ## adv test
        adv_samples = adversary.generate(data, target, **configs1)
        output1 = model(adv_samples)
        #test_loss += F.cross_entropy(output1, target)
        pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct_adv += pred1.eq(target.view_as(pred1)).sum().item()

    print('clean accuracy  = ' + str(correct / len(test_loader.dataset)))
    print('adv accuracy  = ' + str(correct_adv / len(test_loader.dataset)))



def main(args):

    model = assign_model(args.model, 'cuda')
    train_loader, test_loader = feed_dataset('CIFAR10', 'deeprobust/image/data')

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,50,65], gamma=0.2)

    save_model = True
    maxepoch = args.epoch
    device = 'cuda'
    beta = args.beta

    ## attack parameters
    adversary = PGD(model)

    configs1 =  {
    'epsilon': 8/255,
    'num_steps': 20,
    'step_size': 1/855,
    'clip_max': 1,
    'clip_min': 0
    }

    for epoch in range(1, maxepoch + 1):

        train(model, train_loader, optimizer, epoch, beta, device)
        lr_scheduler.step(epoch)

        if (epoch > 1)*(epoch % 1 == 0):
            test(model, test_loader, adversary, configs1, device)

        if (save_model and epoch % 10  == 0):
            if os.path.isdir('base/models/'):
                print('Save model.')
                torch.save(model.state_dict(), 'base/models/'+ 'trade_' +str(epoch) + '_'+
                           str(args.beta) + '.pt')
            else:
                os.mkdir('base/models/')
                print('Make directory and save model.')
                torch.save(model.state_dict(), 'base/models/'+ 'trade_' +str(epoch) + '_'+
                           str(args.beta) + '.pt')





if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=80)
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--beta', help='trade off parameter', default=1.0)
    argparser.add_argument('--model', help='model structure', default='ResNet18')
    args = argparser.parse_args()

    main(args)
