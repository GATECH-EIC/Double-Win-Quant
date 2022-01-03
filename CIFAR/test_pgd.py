import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from model.preact_resnet import PreActResNet18
from model.wide_resnet import WideResNet32
from model.mobilenetv2 import MobileNetV2

from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

from utils import set_random_precision, evaluate_pgd_random, evaluate_standard_random, evaluate_pgd_heter_prec, evaluate_pgd_heter_prec_random

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='/home/yf22/dataset', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--network', default='PreActResNet18', choices=['PreActResNet18', 'WideResNet32', 'MobileNetV2'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')

    parser.add_argument('--num_bits_list', default=None, type=int, nargs='*',
                        help='forward precision list')

    parser.add_argument('--num_attack_bits_list', default=None, type=int, nargs='*',
                        help='attack precision list for switchable precision')

    parser.add_argument('--switchable_bn', action='store_true',
                        help='if enable switchable precision')

    parser.add_argument('--eval_with_single_bit', action='store_true',
                        help='If eval with heterogeneous precision only')

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

    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset)

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

    model.load_state_dict(partial)
    # state = model.state_dict()
    # pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
    # state.update(pretrained_dict)
    # model.load_state_dict(state)

    start_epoch = pretrained_model['epoch'] + 1

    print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)

    if not args.eval_with_single_bit:
        logger.info('Evaluating RPS with standard images...')
        _, test_acc = evaluate_standard_random(test_loader, model, args)
        logger.info('RPS Nature Acc: %.4f \t', test_acc)


        logger.info('Evaluating RPS with PGD-20 Attack...')
        _, pgd_20_acc = evaluate_pgd_heter_prec_random(test_loader, model, 20, 1, logger, args, num_round=1, log_detail=True)
        logger.info('RPS PGD-20 Acc: %.4f \t', pgd_20_acc)


        logger.info('Evaluating RPS with PGD-100 Attack...')
        _, pgd_100_acc = evaluate_pgd_heter_prec_random(test_loader, model, 100, 1, logger, args, num_round=1, log_detail=True)
        logger.info('RPS PGD-100 Acc: %.4f \t', pgd_100_acc)


        logger.info('Evaluating RPS with PGD-20 Attack + Restart 10...')
        _, pgd_20_r10_acc = evaluate_pgd_heter_prec_random(test_loader, model, 20, 10, logger, args, num_round=1, log_detail=True)
        logger.info('RPS PGD-20-Restart-10 Acc: %.4f \t', pgd_20_r10_acc)


        logger.info('RPS Acc: Nature: %.4f \t PGD-20: %.4f \t  PGD-100: %.4f \t PGD-20-Restart-10: %.4f \t', 
                     test_acc, pgd_20_acc, pgd_100_acc, pgd_20_r10_acc)

    
    else:
        logger.info('Evaluating with standard images...')

        nature_acc_array = []
        for num_bits in args.num_bits_list:
            args.num_bits = num_bits
            _, test_acc = evaluate_standard(test_loader, model, args)
            nature_acc_array.append(test_acc)
        
        logger.info('Nature Acc Array: %s', str(nature_acc_array))


        logger.info('Evaluating with PGD-20 under heterogeneous precision...')

        pgd_acc_array = []
        for num_bits in args.num_bits_list:
            args.num_bits = num_bits
            pgd_acc_list = []

            for num_attack_bits in args.num_attack_bits_list:
                args.num_attack_bits = num_attack_bits
                pgd_loss, pgd_acc = evaluate_pgd_heter_prec(test_loader, model, 20, 1, logger, args, log_detail=False)
                logger.info('Bits: %d-bit \t Attack Bits: %d-bit \t PGD Loss: %.4f \t PGD Acc: %.4f \t', args.num_bits, args.num_attack_bits, pgd_loss, pgd_acc)

                pgd_acc_list.append(pgd_acc)

            pgd_acc_array.append(pgd_acc_list)
        
        logger.info('PGD-20 Acc Array: %s', str(pgd_acc_array))


        # logger.info('Evaluating with PGD-100 under heterogeneous precision...')

        # pgd_acc_array = []
        # for num_bits in args.num_bits_list:
        #     args.num_bits = num_bits
        #     pgd_acc_list = []

        #     for num_attack_bits in args.num_attack_bits_list:
        #         args.num_attack_bits = num_attack_bits
        #         pgd_loss, pgd_acc = evaluate_pgd_heter_prec(test_loader, model, 100, 1, logger, args, log_detail=False)
        #         logger.info('Bits: %d-bit \t Attack Bits: %d-bit \t PGD Loss: %.4f \t PGD Acc: %.4f \t', args.num_bits, args.num_attack_bits, pgd_loss, pgd_acc)

        #         pgd_acc_list.append(pgd_acc)

        #     pgd_acc_array.append(pgd_acc_list)
        
        # logger.info('PGD-100 Acc Array: %s', str(pgd_acc_array))


        # logger.info('Evaluating with PGD-20 (Restart 10) under heterogeneous precision...')

        # pgd_acc_array = []
        # for num_bits in args.num_bits_list:
        #     args.num_bits = num_bits
        #     pgd_acc_list = []

        #     for num_attack_bits in args.num_attack_bits_list:
        #         args.num_attack_bits = num_attack_bits
        #         pgd_loss, pgd_acc = evaluate_pgd_heter_prec(test_loader, model, 20, 10, logger, args, log_detail=False)
        #         logger.info('Bits: %d-bit \t Attack Bits: %d-bit \t PGD Loss: %.4f \t PGD Acc: %.4f \t', args.num_bits, args.num_attack_bits, pgd_loss, pgd_acc)

        #         pgd_acc_list.append(pgd_acc)

        #     pgd_acc_array.append(pgd_acc_list)
        
        # logger.info('PGD-20 (Restart 10) Acc Array: %s', str(pgd_acc_array))

    logger.info('Finishing testing.')


if __name__ == "__main__":
    main()
