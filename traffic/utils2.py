import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import PIL.Image as Image


### Data Initialization and Loading
from data import data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation, \
    data_jitter_contrast, data_rotate, data_hvflip, data_shear, data_translate, data_center, data_hflip, \
    data_vflip  # data.py in the same folder

def train_loader(batch_size):
    use_gpu = True
    loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset([datasets.ImageFolder('data/' + 'train/',
                                                         transform=data_transforms),
                                    datasets.ImageFolder('data/'+ 'train/',
                                                         transform=data_jitter_brightness),
                                    datasets.ImageFolder('data/'+ 'train/',
                                                         transform=data_jitter_hue),
                                    datasets.ImageFolder('data/'+ 'train/',
                                                         transform=data_jitter_contrast),
                                    datasets.ImageFolder('data/' + 'train/',
                                                         transform=data_jitter_saturation),
                                    datasets.ImageFolder('data/' + 'train/',
                                                         transform=data_translate),
                                    datasets.ImageFolder('data/' + 'train/',
                                                         transform=data_rotate),
                                    #datasets.ImageFolder('data/'  + '/train_images',
                                    #                     transform=data_hvflip),
                                    datasets.ImageFolder('data/' + 'train/',
                                                         transform=data_center),
                                    datasets.ImageFolder('data/' + 'train/',
                                                         transform=data_shear)]), batch_size=batch_size,
    shuffle=True, num_workers=1, pin_memory=use_gpu)
    return loader

def valid_loader (batch_size):
    use_gpu = True
    loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('data/' + '/validation',
                         transform=data_transforms),
    batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=use_gpu)
    return loader



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



def in_class(predict, label):

    probs = torch.zeros(17)
    for i in range(17):
        in_class_id = torch.tensor(label == i, dtype= torch.float)
        correct_predict = torch.tensor(predict == label, dtype= torch.float)
        in_class_correct_predict = (correct_predict) * (in_class_id)
        acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
        probs[i] = acc

    return probs



def match_weight(label, diff0, diff1, diff2):

    weight0 = torch.zeros(label.shape[0], device='cuda')
    weight1 = torch.zeros(label.shape[0], device='cuda')
    weight2 = torch.zeros(label.shape[0], device='cuda')

    for i in range(17):
        weight0 += diff0[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight1 += diff1[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight2 += diff2[i] * torch.tensor(label == i, dtype= torch.float).cuda()

    weight2 = torch.exp(2 * weight2)
    return weight0, weight1, weight2



def cost_sensitive(lam0, lam1, lam2):

    diff0 = torch.zeros(17)
    for i in range(17):
        for j in range(17):
            if j == i:
                diff0[i] = diff0[i] + 16 / 17 * lam0[i]
            else:
                diff0[i] = diff0[i] - 16 / 17 * lam0[j]
        diff0[i] = diff0[i] + 1 / 17

    diff1 = torch.zeros(17)
    for i in range(17):
        for j in range(17):
            if j == i:
                diff1[i] = diff1[i] + 16 / 17 * lam1[i]
            else:
                diff1[i] = diff1[i] - 16 / 17 * lam1[j]
        diff1[i] = diff1[i] + 1 / 17

    diff2 = lam2

    diff0 = torch.clamp(diff0, min = 0)
    diff1 = torch.clamp(diff1, min = 0)

    return diff0, diff1, diff2



def trades_adv(model,
               x_natural,
               weight2,
               step_size= 3 / 255,
               epsilon= 12 /255,
               num_steps= 6,
               distance='l_inf'
               ):

    new_eps = (epsilon * weight2).view(weight2.shape[0], 1, 1, 1)

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average = False)
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    if distance == 'l_inf':
        for _ in range(num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            eta = step_size * torch.sign(grad.detach())
            eta = torch.min(torch.max(eta, -1.0 * new_eps), new_eps)
            x_adv = x_adv.detach() + eta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv



def train_one_epoch(model, epoch, loader, optimizer, diff0, diff1, diff2, config):

    criterion_kl = nn.KLDivLoss(reduction='none')
    #criterion_kl = nn.KLDivLoss(size_average = False)
    criterion_nat = nn.CrossEntropyLoss(reduction='none')
    #criterion_nat = nn.CrossEntropyLoss()

    print('now epoch:  ' + str(epoch))

    model.train()
    for batch_idx, (data, target) in enumerate(loader):

        data = data.cuda()
        target = target.cuda()

        weight0, weight1, weight2 = match_weight(target, diff0, diff1, diff2)

        ## generate attack
        x_adv = trades_adv(model, data, weight2, **config)

        model.train()
        ## get loss
        loss_natural = criterion_nat(model(data), target)
        loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(data), dim=1))

        # loss_natural_avg = torch.sum(loss_natural * weight0) / torch.sum(weight0)
        loss_robust1 = torch.sum(loss_robust, 1)
        loss = torch.sum(weight0 * loss_natural + 2 * weight1 * loss_robust1) / torch.sum(weight0 + 2 * weight1)

        ## back propagates
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()




def test_valid(model, loader, configs):

    print('Currently Testing the Unfairness on Training Set')
    all_label = []
    all_pred = []
    all_pred_adv = []

    model.eval()
    correct = 0
    correct_adv = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = Variable(data), Variable(target)

        data = data.cuda()
        target = target.cuda()
        all_label.append(target)

        ## clean test
        output = model(data)
        pred = output.max(dim=1)[1]
        all_pred.append(pred)
        add = pred.eq(target.view_as(pred)).sum().item()
        correct += add

        ## generate PGD attack
        adv_samples = pgd_attack(model, data, target, **configs)
        output = model(adv_samples)
        pred_adv = output.max(dim=1)[1]
        all_pred_adv.append(pred_adv)
        add1 = pred_adv.eq(target.view_as(pred_adv)).sum().item()
        correct_adv += add1


    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()

    acc = in_class(all_pred, all_label)
    acc_adv = in_class(all_pred_adv, all_label)

    total_clean_error = 1- correct / all_label.shape[0]
    total_bndy_error = correct / all_label.shape[0] - correct_adv / all_label.shape[0]

    class_clean_error = 1 - acc
    class_bndy_error = acc - acc_adv

    return class_clean_error, class_bndy_error, total_clean_error, total_bndy_error


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def best_model(model, train_loader, optimizer, LayerOneTrainer, diff0, diff1, diff2, epoch, beta, device, rounds):

    criterion_kl = nn.KLDivLoss(reduction='none')
    #criterion_kl = nn.KLDivLoss(size_average = False)
    criterion_nat = nn.CrossEntropyLoss(reduction='none')
    #criterion_nat = nn.CrossEntropyLoss()

    print('now epoch:  ' + str(epoch))
    #pbar.set_description('Trades, Now epoch ' + str(epoch))

    for j in range(rounds):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)

            weight0, weight1, weight2 = match_weight(target, diff0, diff1, diff2)
            ## generate attack
            x_adv = trades_adv(model, data, LayerOneTrainer, weight2)

            model.train()
            ## clear grads
            optimizer.zero_grad()
            LayerOneTrainer.param_optimizer.zero_grad()

            ## get loss
            loss_natural = criterion_nat(model(data), target)

            loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(data), dim=1))
            loss_robust1 = torch.sum(loss_robust, 1)

            loss = torch.sum(weight0 *loss_natural + weight1 * loss_robust1) / torch.sum(weight0 + weight1)

            ## back propagates
            loss.backward()
            optimizer.step()
            LayerOneTrainer.param_optimizer.step()

            ## clear grads
            optimizer.zero_grad()
            LayerOneTrainer.param_optimizer.zero_grad()


