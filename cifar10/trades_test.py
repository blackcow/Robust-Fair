from __future__ import print_function

from utils import *
from tqdm import tqdm
from network import create_network
# from collections import OrderedDict
#

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

    model = assign_model(args.model, 'cuda')
    model.load_state_dict(torch.load('base2/models/'+args.path))

    train_loader, test_loader = feed_dataset('CIFAR10', 'deeprobust/image/data')

    device = 'cuda'

    ## attack parameters during test
    adversary = PGD(model)
    configs1 =  {
    'epsilon': 8/255,
    'num_steps': 10,
    'step_size': 2/255,
    'clip_max': 1,
    'clip_min': 0
    }

    test(model, test_loader, adversary, configs1, device)




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', help='model structure', default='PreResNet18')
    argparser.add_argument('--path', help='model structure', default='trades1.pt')
    args = argparser.parse_args()

    main(args)
