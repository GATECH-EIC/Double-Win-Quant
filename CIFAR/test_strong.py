import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from torchvision.utils import save_image

from model.preact_resnet import PreActResNet18
from model.wide_resnet import WideResNet32
from model.mobilenetv2 import MobileNetV2

from utils import (upper_limit, lower_limit, std, mu, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

from utils import set_random_precision, evaluate_pgd_random, evaluate_standard_random, evaluate_pgd_heter_prec, evaluate_pgd_heter_prec_random

from advertorch.attacks import CarliniWagnerL2Attack
from autoattack import AutoAttack

import torchattacks

from custom_attack.cw_inf import CWLInfAttack

# from foolbox.attacks import L2CarliniWagnerAttack
# from foolbox.criteria import Misclassification

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='/home/yf22/dataset', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--network', default='PreActResNet18', choices=['PreActResNet18', 'WideResNet32', 'MobileNetV2'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')

    parser.add_argument('--attack_type', default='cw_l2', choices=['cw_l2', 'auto_attack', 'cw_inf'])

    parser.add_argument('--num_bits_list', default=None, type=int, nargs='*',
                        help='forward precision list')

    parser.add_argument('--num_attack_bits_list', default=None, type=int, nargs='*',
                        help='attack precision list for switchable precision')

    parser.add_argument('--switchable_bn', action='store_true',
                        help='if enable switchable precision')

    parser.add_argument('--eval_with_single_bit', action='store_true',
                        help='If eval with heterogeneous precision only')

    parser.add_argument('--tau', default=0.1, type=float, help='tau in cw inf')

    parser.add_argument('--max_iterations', default=100, type=int, help='max iterations in cw attack')
    
    parser.add_argument('--c', default=1e-4, type=float, help='c in torchattacks')
    parser.add_argument('--steps', default=1000, type=int, help='steps in torchattacks')

    return parser.parse_args()


def main():
    args = get_args()

    assert type(args.pretrain) == str and os.path.exists(args.pretrain)
    assert args.num_bits_list is not None
    
    if args.num_attack_bits_list is None:
        args.num_attack_bits_list = args.num_bits_list
    
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    logger.info('Inference Bits: %s', args.num_bits_list)
    logger.info('Attack Bits: %s', args.num_attack_bits_list)
    logger.info('Dataset: %s', args.dataset)

    handlers = [logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset, norm=False)

    if args.network == 'PreActResNet18':
        net = PreActResNet18
    elif args.network == 'WideResNet32':
        net = WideResNet32
    elif args.network == 'MobileNetV2':
        net = MobileNetV2
    else:
        print('Wrong network:', args.network)

    if args.switchable_bn:
        model = net(args.num_bits_list, num_classes=args.num_classes, normalize=dataset_normalization).cuda()
    else:
        model = net(num_classes=args.num_classes, normalize=dataset_normalization).cuda()

    model = torch.nn.DataParallel(model)

    pretrained_model = torch.load(args.pretrain)
    partial = pretrained_model['state_dict']

    model.load_state_dict(partial, strict=False)
    # state = model.state_dict()
    # pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
    # state.update(pretrained_dict)
    # model.load_state_dict(state)

    start_epoch = pretrained_model['epoch'] + 1

    print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)

    logger.info('Evaluating RPS with standard images...')
    _, nature_acc = evaluate_standard_random(test_loader, model, args)
    logger.info('RPS Nature Acc: %.4f \t', nature_acc)

    if args.attack_type == 'cw_l2':
        logger.info('Evaluating RPS with CW (L2) attacked images...')

        adversary = CarliniWagnerL2Attack(
            model, num_classes=args.num_classes, confidence=0, binary_search_steps=1, max_iterations=args.max_iterations, initial_const=0.1)  # advtorch
        # adversary = torchattacks.CW(model, c=args.c, kappa=0, steps=args.steps, lr=0.01)  # torchattacks
        # adversary = L2CarliniWagnerAttack(binary_search_steps=1, steps=100, initial_const=0.1) # foolbox

        test_loss = 0
        test_acc = 0
        n = 0
        model.eval()

        test_loader = iter(test_loader)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
        for i in pbar:
            X, y = test_loader.next()
            X, y = X.cuda(), y.cuda()

            model.module.set_precision(num_bits=np.random.choice(args.num_bits_list), num_grad_bits=0)
            
            X_adv = adversary.perturb(X, y)  # advtorch

            # X_adv = adversary(X, y)  # torchattacks

            # X_adv = adversary(X, y, criterion=Misclassification, epsilons=None)  # foolbox

            model.module.set_precision(num_bits=np.random.choice(args.num_bits_list), num_grad_bits=0)

            with torch.no_grad():
                output = model(X_adv)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

            # save_image(X[0], 'saved_img_orig.png')
            # save_image(X_adv[0], 'saved_img_pert.png')
            # print(output.max(1)[1][0], y[0])
            # input()

        cw_l2_acc = test_acc/n

        logger.info('RPS Acc: Nature: %.4f \t CW-L2: %.4f \t', 
                    nature_acc, cw_l2_acc)


    if args.attack_type == 'cw_inf':
        logger.info('Evaluating RPS with CW (Inf) attacked images...')

        adversary = CWLInfAttack(
            model, num_classes=args.num_classes, tau=args.tau, confidence=0, binary_search_steps=1, max_iterations=args.max_iterations, initial_const=0.1)  # advtorch

        test_loss = 0
        test_acc = 0
        n = 0
        model.eval()

        test_loader = iter(test_loader)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
        for i in pbar:
            X, y = test_loader.next()
            X, y = X.cuda(), y.cuda()

            model.module.set_precision(num_bits=np.random.choice(args.num_bits_list), num_grad_bits=0)
            
            X_adv = adversary.perturb(X, y)  # advtorch

            model.module.set_precision(num_bits=np.random.choice(args.num_bits_list), num_grad_bits=0)

            with torch.no_grad():
                output = model(X_adv)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

            print((X_adv-X).abs().max())

            # print((X_adv[0]-X[0]).abs().max())
            # save_image(X[0], 'saved_img_orig.png')
            # save_image(X_adv[0], 'saved_img_pert.png')
            # print(output.max(1)[1][0], y[0])
            # input()

        cw_inf_acc = test_acc/n

        logger.info('RPS Acc: Nature: %.4f \t CW-Inf: %.4f \t', 
                    nature_acc, cw_inf_acc)



    if args.attack_type == 'auto_attack':
        logger.info('Evaluating RPS with Auto Attack images...')

        epsilon = args.epsilon / 255
        adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')

        test_loss = 0
        test_acc = 0
        n = 0
        model.eval()

        test_loader = iter(test_loader)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)

        for i in pbar:
            X, y = test_loader.next()
            X, y = X.cuda(), y.cuda()

            model.module.set_precision(num_bits=np.random.choice(args.num_bits_list), num_grad_bits=0)

            X_adv = adversary.run_standard_evaluation(X, y, bs=args.batch_size)

            model.module.set_precision(num_bits=np.random.choice(args.num_bits_list), num_grad_bits=0)

            with torch.no_grad():
                output = model(X_adv)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

        auto_attack_acc = test_acc/n

        logger.info('RPS Acc: Nature: %.4f \t Auto Attack: %.4f \t', 
                        nature_acc, auto_attack_acc)


    logger.info('Finishing testing.')


if __name__ == "__main__":
    main()
