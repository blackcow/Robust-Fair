# 针对 wideResNet 的 FRL
# 基于 TRADES 的 pre-train model
# ../../Fair-AT/model-cifar-wideResNet/wideresnet/TRADES/e0.031_depth34_widen10_drop0.0/model-wideres-epoch76.pt
from __future__ import print_function

# from utils1 import *
from utils1_wd import *
# from network import create_network
from preactresnet import create_network
from wideresnet import *
from data_loader import get_cifar10_loader

import os
import time
import argparse
import torch.optim as optim
import torch.nn as nn
import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def assign_model(model, device='cuda'):
    if (model == 'WideResNet'):
        train_net = nn.DataParallel(WideResNet().cuda())
    else:
        raise ValueError('have no satisfied model.')
    return train_net


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    path = './models-wideresnet/fair1/'
    if not os.path.isdir(path):
        os.makedirs(path)
    logger = get_logger(path + 'train.log')
    h_net = assign_model(args.model, 'cuda')
    ds_train, ds_valid, ds_test = get_cifar10_loader(batch_size=args.batch_size)

    if args.hot == 1:  # based on pre-trained model (AT: TRADES)
        h_net.load_state_dict(torch.load('../../Fair-AT/model-cifar-wideResNet/wideresnet/'
                                         'TRADES/e0.031_depth34_widen10_drop0.0/model-wideres-epoch76.pt'))
        lr = 0.001
        ms = [80, 100, 120]
    else:
        lr = 0.01
        ms = [40, 80, 120]

    ## other layer optimizer
    # optimizer = optim.SGD(h_net.other_layers.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(h_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=0.2)

    # ## layer one optimizer
    Hamiltonian_func = Hamiltonian(h_net, 5e-4)
    # layer_one_optimizer = optim.SGD(h_net.layer_one.parameters(), lr=lr_scheduler.get_lr()[0], momentum=0.9,
    #                                 weight_decay=5e-4)
    # layer_one_optimizer_lr_scheduler = optim.lr_scheduler.MultiStepLR(layer_one_optimizer,
    #                                                                   milestones=ms, gamma=0.2)
    LayerOneTrainer = FastGradientLayerOneTrainer(Hamiltonian_func, optimizer,
                                                  args.inner_iters, sigma=2 / 255, eps=8 / 255)

    maxepoch = args.epoch
    device = 'cuda'
    beta = args.beta

    ############################### initialize exponentional gradient setting
    rate = args.rate
    delta0 = args.bound0 * torch.ones(10)
    delta1 = args.bound1 * torch.ones(10)
    lmbda = torch.zeros(30)

    ## attack parameters during test
    configs1 = {'epsilon': 8 / 255, 'num_steps': 10, 'step_size': 2 / 255, 'clip_max': 1, 'clip_min': 0}

    ## record the results
    valid_clean_avg = []
    valid_adv_avg = []
    valid_clean_wst = []
    valid_adv_wst = []

    test_clean_avg = []
    test_adv_avg = []
    # the worst of benign & robust acc
    test_clean_wst = []
    test_adv_wst = []

    Gamma0 = []
    Gamma1 = []

    ### main training loop
    for now_epoch in range(1, maxepoch + 1):
        start = time.time()
        ## doing evaluation on test data
        a1, a2, a3, a4 = evaluate(h_net, ds_test, configs1, device, logger)
        ## record the results for test set
        test_clean_avg.append(a1)
        test_adv_avg.append(a2)
        test_clean_wst.append(np.min(a3))
        test_adv_wst.append(np.min(a4))

        # print('train epoch ' + str(now_epoch), flush=True)
        logger.info('train epoch ' + str(now_epoch))


        ## given model, get the validation performance and gamma
        class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
            best_lambda(h_net, ds_valid, configs1, device, logger)

        ## get gamma on validation set
        gamma0 = class_clean_error - total_clean_error - delta0
        gamma1 = class_bndy_error - total_bndy_error - delta1

        Gamma0.append(torch.max(gamma0).item())
        Gamma1.append(torch.max(gamma1).item())

        ## print inequality results
        # print('current results: clean error, boundary error')
        # print('total clean error ' + str(total_clean_error))
        # print('total boundary error ' + str(total_bndy_error))
        # logger.info('total clean error ' + str(total_clean_error))
        # logger.info('total boundary error ' + str(total_bndy_error))

        logger.info('total clean error:{:.3f}'.format(total_clean_error))
        logger.info('total boundary error:{:.3f}'.format(total_bndy_error))

        # print('each class errors')
        # print('clean_error', class_clean_error)
        # print('bndy_error', class_bndy_error)

        logger.info('each class errors')
        logger.info('clean_error', str(class_clean_error))
        logger.info('bndy_error', str(class_bndy_error))

        # print('.............')
        # print('each class inequality constraints')
        # print(class_clean_error - total_clean_error)
        # print(class_bndy_error - total_bndy_error)
        logger.info('.............')
        logger.info('each class inequality constraints')
        logger.info(class_clean_error - total_clean_error)
        logger.info(class_bndy_error - total_bndy_error)

        ## record the results for validation set
        valid_adv_avg.append(1 - (total_bndy_error + total_clean_error))
        valid_clean_avg.append(1 - total_clean_error)
        valid_adv_wst.append(1 - torch.max((class_bndy_error + class_clean_error)).item())
        valid_clean_wst.append(1 - torch.max(class_clean_error).item())

        #################################################### do training on now epoch
        ## update theta to do reweight / re-epsilon
        if rate % 50 == 0:
            rate = rate / 2

        ## constraints coefficients
        lmbda0 = lmbda[0:10] + rate * (gamma0)
        lmbda0 = torch.clamp(lmbda0, min=0)

        lmbda1 = lmbda[10:20] + args.rate2 * rate * (gamma1)
        lmbda1 = torch.clamp(lmbda1, min=0)

        lmbda2 = lmbda[20:30] + args.rate2 * rate * (gamma1)
        lmbda2 = torch.clamp(lmbda2, min=-0.1)

        lmbda = torch.cat([lmbda0, lmbda1, lmbda2])

        ## given best lambda, solving outside to find best model
        # print('..............................')
        logger.info('..............................')
        diff0, diff1, diff2 = cost_sensitive(lmbda0, lmbda1, lmbda2, beta)

        # print('current weight')
        # print(diff0)
        # print(diff1)
        # print(diff2)
        # print('..............................')

        logger.info('current weight')
        logger.info(diff0)
        logger.info(diff1)
        logger.info(diff2)
        logger.info('..............................')

        ## do the model parameter update based on gamma
        _ = best_model(h_net, ds_train, optimizer, LayerOneTrainer, diff0, diff1, diff2, now_epoch,
                       beta, device, rounds=args.inner_epoch, logger=logger)
        lr_scheduler.step(now_epoch)
        # layer_one_optimizer_lr_scheduler.step(now_epoch)

        # print('................................................................................')
        # print('................................................................................')
        # print('................................................................................')
        # print('test on current epoch')
        logger.info('................................................................................')
        logger.info('................................................................................')
        logger.info('................................................................................')
        logger.info('test on current epoch')

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

        if not os.path.isdir(path):
            os.mkdir(path)

        record_name = path + 'rwre_' + str(args.bound0) + '_' + str(args.bound1)
        np.savetxt(record_name, table1)

        end = time.time()
        tm = (end - start) / 60
        # print('时间(分钟):' + str(tm))
        logger.info('时间(分钟):' + str(tm))

        if (now_epoch % 10 == 0):
            ## save model
            torch.save(h_net.state_dict(), path + 'trade_' + str(now_epoch) + '_' +
                           str(args.beta) + '.pt')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=120)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=200)
    argparser.add_argument('--seed', type=int, help='random seed', default=100)
    argparser.add_argument('--beta', help='trade off parameter', type=float, default=1.0)
    argparser.add_argument('--model', help='model structure', default='PreResNet18')
    argparser.add_argument('--path', help='model path', default='trades1.pt')
    argparser.add_argument('--inner_iters', type=int, default=4)
    argparser.add_argument('--bound0', type=float, help='fair constraints for clean error', default=0.05)
    argparser.add_argument('--bound1', type=float, help='fair constraints for adv error', default=0.05)
    argparser.add_argument('--rate', type=float, help='hyper-par update rate', default=0.2)
    argparser.add_argument('--inner_epoch', type=int, help='inner rounds', default=1)
    argparser.add_argument('--hot', type=int, help='whether hot start', default=0)
    argparser.add_argument('--rate2', type=float, help='hyper-par update rate', default=1.0)
    argparser.add_argument('--gpu-id', type=str, default='0,1', help='gpu_id')
    args = argparser.parse_args()

    main(args)
