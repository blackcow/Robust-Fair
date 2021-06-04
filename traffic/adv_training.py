


from deeprobust.image.attack.pgd import PGD
import torch
from torchvision import datasets, transforms
from deeprobust.image.config import defense_params
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse


"""
LOAD DATASETS
"""

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

    return train_net


if ('CIFAR10' == 'CIFAR10'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10( 'deeprobust/image/data', train=True, download=True,
                         transform=transform_train),
        batch_size=100, shuffle=True)  # , **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(  'deeprobust/image/data', train=False, download=True,
                         transform=transform_val),
        batch_size=100, shuffle=True)  # , **kwargs)



save_model = True
configs = {
    'epsilon': 8/255,
    'num_steps': 12,
    'step_size': 1.2/255,
    'clip_max': 1,
    'clip_min': 0
}

configs1 = {
    'epsilon': 8/255,
    'num_steps': 20,
    'step_size': 0.8/255,
    'clip_max': 1,
    'clip_min': 0
}



"""
TRAIN DEFENSE MODEL
"""

def main(args):

    torch.manual_seed(args.seed)
    model = assign_model(args.model, 'cuda')
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 50, 60], gamma=0.2)

    for epoch in range(1, 60 + 1):  ## 5 batches

        print('Epoch ' + str(epoch))

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = torch.tensor(data).to('cuda'), torch.tensor(target).to('cuda')

            ## generate PGD attack
            adversary = PGD(model)
            adv_samples = adversary.generate(data, target, **configs)

            output = model(adv_samples)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 5 == 1:

            correct = 0
            acc_correct = 0

            for data, target in test_loader:

                ## clean acc
                data, target = data.to('cuda'), target.to('cuda')
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
                correct  = correct + torch.sum(pred == target)

                ## adv acc
                adversary = PGD(model)
                adv_samples = adversary.generate(data, target, **configs1)
                output = model(adv_samples)

                pred1 = output.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
                acc_correct  =  acc_correct + torch.sum(pred1 == target)

            correct = correct.item() / len(test_loader.dataset)
            acc_correct = acc_correct.item() / len(test_loader.dataset)

            print('clean accuracy = ' + str(correct))
            print('adv accuracy = ' + str(acc_correct))

        if (save_model and epoch % 10 == 1):
            if os.path.isdir('transfer/robust_models/'):
                print('Save model.')
                torch.save(model.state_dict(),
                       'transfer/robust_models/' + args.model + '_' + str(args.seed)+ '.pt')
            else:
                os.mkdir('transfer/robust_models/')
                print('Make directory and save model.')
                torch.save(model.state_dict(),
                           'transfer/robust_models/' + args.model + '_'+ str(args.seed) + '.pt')

    print('====== FINISH TRAINING =====')




if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', help='victim model', default='ResNet18')
    argparser.add_argument('--seed', type= int, default='100')
    args = argparser.parse_args()

    main(args)