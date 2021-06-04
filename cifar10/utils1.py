import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from torchvision import datasets, transforms
import numpy as np
from deeprobust.image.attack.pgd import PGD



class Hamiltonian(_Loss):

    def __init__(self, layer, reg_cof = 1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0


    def forward(self, x, p):

        y = self.layer(x)
        H = torch.sum(y * p)
        return H


def cal_l2_norm(layer: torch.nn.Module):
 loss = 0.
 for name, param in layer.named_parameters():
     if name == 'weight':
         loss = loss + 0.5 * torch.norm(param,) ** 2

 return loss

class FastGradientLayerOneTrainer(object):

    def __init__(self, Hamiltonian_func, param_optimizer,
                    inner_steps=2, sigma = 0.008, eps = 0.03):
        self.inner_steps = inner_steps
        self.sigma = sigma
        self.eps = eps
        self.Hamiltonian_func = Hamiltonian_func
        self.param_optimizer = param_optimizer

    def step(self, inp, p, eta, weight):

        p = p.detach()
        new_eps = (self.eps * weight).view(weight.shape[0], 1, 1, 1)

        for i in range(self.inner_steps):
            tmp_inp = inp + eta
            tmp_inp = torch.clamp(tmp_inp, 0, 1)
            H = self.Hamiltonian_func(tmp_inp, p)

            eta_grad = torch.autograd.grad(H, eta, only_inputs=True, retain_graph=False)[0]
            eta_grad_sign = eta_grad.sign()
            eta = eta - eta_grad_sign * self.sigma

            eta = torch.min(torch.max(eta, -1.0 * new_eps), new_eps)
            eta = torch.clamp(inp + eta, 0.0, 1.0) - inp
            eta = eta.detach()
            eta.requires_grad_()
            eta.retain_grad()

        yofo_inp = eta + inp
        yofo_inp = torch.clamp(yofo_inp, 0, 1)

        loss = -1.0 * (self.Hamiltonian_func(yofo_inp, p) -
                       5e-4 * cal_l2_norm(self.Hamiltonian_func.layer))

        loss.backward()

        return yofo_inp, eta



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
                 batch_size= 200, shuffle=True) #, **kwargs)

        test_loader  = torch.utils.data.DataLoader(
                 datasets.CIFAR10(data_dict, train=False, download = True,
                        transform=transform_val),
                batch_size= 100, shuffle=True) #, **kwargs)


    return train_loader, test_loader



def yopo_trades_adv(model,
                    x_natural,
                    LayerOneTrainer,
                    weight,
                    K = 3):

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average = False)

    model.eval()
    eta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
    eta.requires_grad_()

    raw_soft_label = F.softmax(model(x_natural), dim=1).detach()
    for j in range(K):
        pred = model(x_natural + eta.detach())
        with torch.enable_grad():
            loss = criterion_kl(F.log_softmax(pred, dim=1), raw_soft_label)  # raw_soft_label.detach())

        p = -1.0 * torch.autograd.grad(loss, [model.layer_one_out, ])[0]
        yofo_inp, eta = LayerOneTrainer.step(x_natural, p, eta, weight)

    x_adv = x_natural + eta
    return x_adv



def in_class(predict, label):

    probs = torch.zeros(10)
    for i in range(10):
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

    for i in range(10):
        weight0 += diff0[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight1 += diff1[i] * torch.tensor(label == i, dtype= torch.float).cuda()
        weight2 += diff2[i] * torch.tensor(label == i, dtype= torch.float).cuda()

    weight2 = torch.exp(2 * weight2)
    return weight0, weight1, weight2



def cost_sensitive(lam0, lam1, lam2, beta):

    diff0 = torch.zeros(10)
    for i in range(10):
        for j in range(10):
            if j == i:
                diff0[i] = diff0[i] + 9 / 10 * lam0[i]
            else:
                diff0[i] = diff0[i] - 1 / 10 * lam0[j]
        diff0[i] = diff0[i] + 1 / 10

    diff1 = torch.zeros(10)
    for i in range(10):
        for j in range(10):
            if j == i:
                diff1[i] = diff1[i] + 9 / 10 * lam1[i]
            else:
                diff1[i] = diff1[i] - 1 / 10 * lam1[j]
        diff1[i] = diff1[i] + beta * 1 / 10

    diff2 = lam2

    diff0 = torch.clamp(diff0, min = 0)
    diff1 = torch.clamp(diff1, min = 0)

    return diff0, diff1, diff2



def best_lambda(model, test_loader, configs1, device):

    print('Doing test on validation set')
    model.eval()

    correct = 0
    correct_adv = 0
    adversary = PGD(model)

    all_label = []
    all_pred = []
    all_pred_adv = []

    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        all_label.append(target)

        ## clean test
        output = model(data)
        #test_loss += F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add = pred.eq(target.view_as(pred)).sum().item()
        correct += add
        model.zero_grad()
        all_pred.append(pred)

        ## adv test
        adv_samples = adversary.generate(data, target, **configs1)
        output1 = model(adv_samples)
        #test_loss += F.cross_entropy(output1, target)
        pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add1 = pred1.eq(target.view_as(pred1)).sum().item()
        correct_adv += add1
        all_pred_adv.append(pred1)

    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()

    acc = in_class(all_pred, all_label)
    acc_adv = in_class(all_pred_adv, all_label)

    total_clean_error = 1- correct / len(test_loader.dataset)
    total_bndy_error = correct / len(test_loader.dataset) - correct_adv / len(test_loader.dataset)

    class_clean_error = 1 - acc
    class_bndy_error = acc - acc_adv

    return class_clean_error, class_bndy_error, total_clean_error, total_bndy_error





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
            x_adv = yopo_trades_adv(model, data, LayerOneTrainer, weight2, K=3)

            model.train()
            ## clear grads
            optimizer.zero_grad()
            LayerOneTrainer.param_optimizer.zero_grad()

            ## get loss
            loss_natural = criterion_nat(model(data), target)
            loss_robust = criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(data), dim=1))

            # loss_natural_avg = torch.sum(loss_natural * weight0) / torch.sum(weight0)
            loss_robust1 = torch.sum(loss_robust, 1)

            #weight0 = weight0 / torch.sum(weight0)
            #weight1 = weight1 / torch.sum(weight1) * beta

            loss = torch.sum(weight0 *loss_natural + weight1 * loss_robust1) / torch.sum(weight0 + weight1)

            ## back propagates
            loss.backward()
            optimizer.step()
            LayerOneTrainer.param_optimizer.step()

            ## clear grads
            optimizer.zero_grad()
            LayerOneTrainer.param_optimizer.zero_grad()




def evaluate(model, test_loader, configs1, device):

    print('Doing test')
    model.eval()

    correct = 0
    correct_adv = 0
    adversary = PGD(model)

    all_label = []
    all_pred = []
    all_pred_adv = []

    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        all_label.append(target)

        ## clean test
        output = model(data)
        #test_loss += F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add = pred.eq(target.view_as(pred)).sum().item()
        correct += add
        model.zero_grad()
        all_pred.append(pred)

        ## adv test
        adv_samples = adversary.generate(data, target, **configs1)
        output1 = model(adv_samples)
        #test_loss += F.cross_entropy(output1, target)
        pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add1 = pred1.eq(target.view_as(pred1)).sum().item()
        correct_adv += add1
        all_pred_adv.append(pred1)

    print('clean accuracy  = ' + str(correct / len(test_loader.dataset)), flush= True)
    print('adv accuracy  = ' + str(correct_adv / len(test_loader.dataset)), flush=True)

    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()

    acc = in_class(all_pred, all_label).numpy()
    acc_adv = in_class(all_pred_adv, all_label).numpy()

    print('each classes clean and adversarial accuracy')
    print(acc)
    print(acc_adv)

    acc_all = correct / len(test_loader.dataset)
    acc_adv_all = correct_adv / len(test_loader.dataset)

    return acc_all, acc_adv_all, acc, acc_adv








