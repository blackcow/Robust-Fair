from __future__ import print_function

from utils1 import *
from preactresnet import create_network
from data_loader import get_cifar10_loader

import os
import argparse
import torch.optim as optim

def assign_model(model, device = 'cuda'):

    if (model == 'PreResNet18'):
        train_net = create_network().cuda()
        #import deeprobust1.image.netmodels.resnet as MODEL
        #train_net = MODEL.ResNet18().to(device)
    elif (model == 'ResNet34'):
        import deeprobust1.image.netmodels.resnet as MODEL
        train_net = MODEL.ResNet34().to(device)

    return train_net




def main(args):

    h_net = assign_model(args.model, 'cuda')
    ds_train, ds_valid, ds_test = get_cifar10_loader(batch_size=args.batch_size)

    if args.hot == 1:
        h_net.load_state_dict(torch.load('base4/models/'+str(args.path)))
        lr = 0.001
        ms = [80, 100, 120]
    else:
        lr = 0.01
        ms = [40, 80, 120]

    ## other layer optimizer
    optimizer = optim.SGD(h_net.other_layers.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.2)

    ## layer one optimizer
    Hamiltonian_func = Hamiltonian(h_net.layer_one, 5e-4)
    layer_one_optimizer = optim.SGD(h_net.layer_one.parameters(), lr=lr_scheduler.get_lr()[0], momentum=0.9,
                                    weight_decay=5e-4)
    layer_one_optimizer_lr_scheduler = optim.lr_scheduler.MultiStepLR(layer_one_optimizer,
                                                                      milestones=ms, gamma=0.2)
    LayerOneTrainer = FastGradientLayerOneTrainer(Hamiltonian_func, layer_one_optimizer,
                                                  args.inner_iters, sigma=2/255, eps=8/255)

    maxepoch = args.epoch
    device = 'cuda'
    beta = args.beta

    ############################### initialize exponentional gradient setting
    rate = args.rate
    delta0 = args.bound0 * torch.ones(10)
    delta1 = args.bound1 * torch.ones(10)
    lmbda = torch.zeros(30)

    ## attack parameters during test
    configs1 =  {
    'epsilon': 8/255,
    'num_steps': 10,
    'step_size': 2/255,
    'clip_max': 1,
    'clip_min': 0
    }

    ## record the results
    valid_clean_avg = []
    valid_adv_avg = []
    valid_clean_wst = []
    valid_adv_wst = []

    test_clean_avg = []
    test_adv_avg = []
    test_clean_wst = []
    test_adv_wst = []

    Gamma0 = []
    Gamma1 = []

    ### main training loop
    for now_epoch in range(1, maxepoch + 1):
        ## doing evaluation
        a1, a2, a3, a4 = evaluate(h_net, ds_test, configs1, device)
        ## record the results for test set
        test_adv_avg.append(a2)
        test_clean_avg.append(a1)
        test_adv_wst.append(np.min(a4))
        test_clean_wst.append(np.min(a3))

        print('train epoch ' + str(now_epoch), flush=True)

        ## given model, get the validation performance and gamma
        class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
            best_lambda(h_net, ds_valid, configs1, device)

        ## get gamma on validation set
        gamma0 = class_clean_error - total_clean_error - delta0
        gamma1 = class_bndy_error - total_bndy_error - delta1

        Gamma0.append(torch.max(gamma0).item())
        Gamma1.append(torch.max(gamma1).item())

        ## print inequality results
        #print('current results: clean error, boundary error')
        print('total clean error ' + str(total_clean_error))
        print('total boundary error ' + str(total_bndy_error))

        print('each class errors')
        print(class_clean_error)
        print(class_bndy_error)

        print('.............')
        print('each class inequality constraints')
        print(class_clean_error - total_clean_error)
        print(class_bndy_error - total_bndy_error)

        ## record the results for validation set
        valid_adv_avg.append(1 - (total_bndy_error+ total_clean_error))
        valid_clean_avg.append(1 - total_clean_error)
        valid_adv_wst.append(1 - torch.max((class_bndy_error + class_clean_error)).item())
        valid_clean_wst.append(1 - torch.max(class_clean_error).item())

        #################################################### do training on now epoch
        ## update theta to do reweight / re-epsilon
        if rate % 50 == 0:
            rate = rate / 2

        ## constraints coefficients
        lmbda0 = lmbda[0:10] + rate * (gamma0)
        lmbda0 = torch.clamp(lmbda0, min = 0 )

        lmbda1 = lmbda[10:20] #+ args.rate2 * rate * (gamma1)
        lmbda1 = torch.clamp(lmbda1, min = 0)

        lmbda2 = lmbda[20:30] + args.rate2 * rate * (gamma1)
        lmbda2 = torch.clamp(lmbda2, min = - 0.1)

        lmbda = torch.cat([lmbda0, lmbda1, lmbda2])

        ## given best lambda, solving outside to find best model
        print('..............................')
        diff0, diff1, diff2 = cost_sensitive(lmbda0, lmbda1, lmbda2, beta)

        print('current weight')
        print(diff0)
        print(diff1)
        print(diff2)
        print('..............................')

        ## do the model parameter update based on gamma
        _ = best_model(h_net, ds_train, optimizer, LayerOneTrainer, diff0, diff1, diff2, now_epoch,
                       beta, device, rounds= args.inner_epoch)
        lr_scheduler.step(now_epoch)
        layer_one_optimizer_lr_scheduler.step(now_epoch)

        print('................................................................................')
        print('................................................................................')
        print('................................................................................')
        print('test on current epoch')

        ##save the results
        TEST1 = np.array(test_adv_avg)
        TEST2 = np.array(test_clean_avg)
        TEST3 = np.array(test_adv_wst)
        TEST4 = np.array(test_clean_wst)

        VALID1 = np.array(valid_adv_avg)
        VALID2 = np.array(valid_clean_avg)
        VALID3 = np.array(valid_adv_wst)
        VALID4 = np.array(valid_clean_wst)

        GG0 = np.array(Gamma0)
        GG1 = np.array(Gamma1)

        table1 = np.stack((VALID1, VALID2, VALID3, VALID4, TEST1, TEST2, TEST3, TEST4, GG0, GG1))

        record_name = 're_' + str(args.bound0) + '_' + str(args.bound1)
        np.savetxt(record_name, table1)


        if (now_epoch % 5  == 0):
            ## save model
            if os.path.isdir('base4/models/'):
                print('Save model.')
                torch.save(h_net.state_dict(), 'base4/models/'+ 'trade_' +str(now_epoch) + '_'+
                           str(args.beta) + '.pt')
            else:
                os.mkdir('base4/models/')
                print('Make directory and save model.')
                torch.save(h_net.state_dict(), 'base4/models/'+ 'trade_' +str(now_epoch) + '_'+
                           str(args.beta) + '.pt')






if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=120)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=200)
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--beta', help='trade off parameter', type = float, default=1.0)
    argparser.add_argument('--model', help='model structure', default='PreResNet18')
    argparser.add_argument('--path', help='model path', default='trades1.pt')
    argparser.add_argument('--inner_iters', type = int, default= 4)
    argparser.add_argument('--bound0', type=float, help='fair constraints for clean error', default=0.05)
    argparser.add_argument('--bound1', type=float, help='fair constraints for adv error', default=0.05)
    argparser.add_argument('--rate', type=float, help='hyper-par update rate', default=0.2)
    argparser.add_argument('--inner_epoch', type=int, help='inner rounds', default=1)
    argparser.add_argument('--hot', type= int, help='whether hot start', default= 1)
    argparser.add_argument('--rate2', type=float, help='hyper-par update rate', default=1.0)
    args = argparser.parse_args()

    main(args)