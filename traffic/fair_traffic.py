from __future__ import print_function

from utils1 import *
import os
import argparse
import torch.optim as optim

def main(args):

    use_gpu = True
    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
    Tensor = FloatTensor

    from model import Net
    model = Net()
    if use_gpu:
        model.cuda()
    model.load_state_dict(torch.load('models/nadv_model_40.pth'))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 50], gamma=0.2)

    maxepoch = args.epoch
    device = 'cuda'

    ds_train = train_loader(args.batch_size)
    ds_valid = valid_loader(args.batch_size)

    configs = {
        'epsilon': 12 / 255,
        'num_steps': 6,
        'step_size': 4 / 255,
    }

    configs1 = {
        'epsilon': 12 / 255,
        'num_steps': 6,
        'step_size': 4 / 255,
        'clip_max': 1,
        'clip_min': 0,
        'print_process': False
    }

    ############################### initialize exponentional gradient setting
    rate = args.rate
    delta0 = args.bound0 * torch.ones(17)
    delta1 = args.bound1 * torch.ones(17)
    lmbda = torch.zeros(17 * 3)

    ### main training loop
    for now_epoch in range(1, maxepoch + 1):

        print('train epoch ' + str(now_epoch), flush=True)

        ## given model, get the validation performance and gamma
        class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
            test_valid(model, ds_valid, configs1)

        ## get gamma on validation set
        gamma0 = class_clean_error - total_clean_error - delta0
        gamma1 = class_bndy_error - total_bndy_error - delta1

        ## print inequality results
        #print('current results: clean error, boundary error')
        print('total clean error ' + str(total_clean_error))
        print('total boundary error ' + str(total_bndy_error))

        print('each class errors')
        print(class_clean_error)
        print(class_bndy_error)

        print('.............')
        print('each class inequality constraints')
        print(torch.max(class_clean_error - total_clean_error))
        print(torch.max(class_bndy_error - total_bndy_error))

        #################################################### do training on now epoch
        ## update theta to do reweight / re-epsilon
        if rate % 30 == 0:
            rate = rate / 2

        ## constraints coefficients
        lmbda0 = lmbda[0:17] + rate / 5 * torch.clamp(gamma0, min = 0)
        lmbda0 = torch.clamp(lmbda0, min = 0)

        lmbda1 = lmbda[17:34]  + rate / 5 * torch.clamp(gamma1, min = 0)
        lmbda1 = torch.clamp(lmbda1, min = 0)
        print(lmbda1)

        lmbda2 = lmbda[34:51] +   2 *  rate * (gamma1)
        lmbda2 = torch.clamp(lmbda2, min = -0.2, max = 0.4)
        print(lmbda2)

        lmbda = torch.cat([lmbda0, lmbda1, lmbda2])

        ## given best lambda, solving outside to find best model
        print('..............................')
        diff0, diff1, diff2 = cost_sensitive(lmbda0, lmbda1, lmbda2)

        ## do the model parameter update based on gamma
        train_one_epoch(model, now_epoch, ds_train, optimizer, diff0, diff1, diff2, configs)
        lr_scheduler.step(now_epoch)

        ## save the model
        if (now_epoch % 1  == 0):
            ## save model
            if os.path.isdir('models/'):
                #print('Save model.')
                torch.save(model.state_dict(), 'models/'+ 'fair_' +str(now_epoch) + '.pt')
            else:
                os.mkdir('models/')
                #print('Make directory and save model.')
                torch.save(model.state_dict(), 'models/'+ 'fair_' +str(now_epoch) + '.pt')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=120)
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=200)
    argparser.add_argument('--bound0', type=float, help='fair constraints for clean error', default=0.1)
    argparser.add_argument('--bound1', type=float, help='fair constraints for adv error', default=0.15)
    argparser.add_argument('--rate', type=float, help='hyper-par update rate', default=0.02)
    argparser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    args = argparser.parse_args()

    main(args)
