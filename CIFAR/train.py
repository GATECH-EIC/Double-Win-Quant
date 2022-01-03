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

from utils import set_random_precision, evaluate_pgd_random, evaluate_standard_random

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='/home/yf22/dataset', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    # parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--epochs', default=160, type=int)
    parser.add_argument('--network', default='PreActResNet18', choices=['PreActResNet18', 'WideResNet32', 'MobileNetV2', 'VGG16'])
    parser.add_argument('--lr_schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr_min', default=0., type=float)
    # parser.add_argument('--lr_max', default=0.2, type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta_init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early_stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--opt_level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss_scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--enable_apex', action='store_true',
        help='If enable apex for half precision training')
    parser.add_argument('--master_weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    
    parser.add_argument('--attack_type', default='fgsm', choices=['fgsm', 'pgd'])

    parser.add_argument('--attack_iters', default=7, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)

    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')

    parser.add_argument('--test_only', action='store_true', help='directly test without training')

    parser.add_argument('--num_bits',default=None,type=int,
                        help='num bits for weight and activation')

    parser.add_argument('--random_precision_training', action='store_true',
                        help='enble random precision training')
    parser.add_argument('--switchable_bn', action='store_true',
                        help='if use switchable BN')               
    parser.add_argument('--num_bits_schedule', default=None, type=int, nargs='*',
                        help='precision schedule for weight/act in random precision training')

    return parser.parse_args()


def main():
    args = get_args()

    if args.attack_type == 'pgd':
        args.alpha = 2

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    
    args.save_dir = os.path.join('ckpt', args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    if args.test_only:
        log_path = os.path.join(args.save_dir, 'output_test.log')
    else:
        log_path = os.path.join(args.save_dir, 'output.log')

    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]

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

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std

    if args.attack_type == 'pgd':
        args.delta_init = 'random'

    if args.network == 'PreActResNet18':
        net = PreActResNet18
    elif args.network == 'WideResNet32':
        net = WideResNet32
    elif args.network == 'MobileNetV2':
        net = MobileNetV2
    else:
        print('Wrong network:', args.network)

    if args.switchable_bn:
        assert args.num_bits_schedule is not None

        num_bits_list = list(range(args.num_bits_schedule[0], args.num_bits_schedule[1]+1))
        model = net(num_bits_list, num_classes=args.num_classes, normalize=dataset_normalization).cuda()
    else:
        model = net(num_classes=args.num_classes, normalize=dataset_normalization).cuda()

    model = torch.nn.DataParallel(model)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.enable_apex:
        amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
        if args.opt_level == 'O2':
            amp_args['master_weights'] = args.master_weights
        model, opt = amp.initialize(model, opt, **amp_args)

    criterion = nn.CrossEntropyLoss()

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    if args.num_bits is None:
        args.num_bits = 0

    prev_robust_acc = 0.
    best_pgd_acc = 0
    test_acc_best_pgd = 0

    if type(args.pretrain) == str and os.path.exists(args.pretrain):
        pretrained_model = torch.load(args.pretrain)
        partial = pretrained_model['state_dict']

        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)

        opt.load_state_dict(pretrained_model['opt'])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        start_epoch = pretrained_model['epoch'] + 1

        best_pgd_acc = pretrained_model['best_pgd_acc']
        test_acc_best_pgd = pretrained_model['standard_acc']

        print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)

    else:
        start_epoch = 0
        print('No checkpoint. Train from scratch.')


    if args.test_only:
        logger.info('Evaluating with PGD Attack...')
        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 20, 1, logger, args)

        logger.info('Evaluating with standard images...')
        test_loss, test_acc = evaluate_standard(test_loader, model, args)

        logger.info('Test Loss: %.4f  \t Test Acc: %.4f  \n PGD Loss: %.4f \t PGD Acc: %.4f \n', 
                    test_loss, test_acc, pgd_loss, pgd_acc)

        logger.info('Finishing testing.')
        exit()

    start_train_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            _iters = epoch * len(train_loader) + i

            if args.random_precision_training:
                set_random_precision(args, _iters, logger)
            
            model.module.set_precision(args.num_bits, 0)

            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True

            if args.attack_type == 'fgsm':
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)

                if args.enable_apex:
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            
            elif args.attack_type == 'pgd':
                for _ in range(args.attack_iters):
                    output = model(X + delta)
                    loss = criterion(output, y)

                    if args.enable_apex:
                        with amp.scale_loss(loss, opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    grad = delta.grad.detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()
            
            else:
                print('Wrong attack type:', args.attack_type)
                exit()
            
            delta = delta.detach()
            
            model.module.set_precision(args.num_bits, 0)

            output = model(X + delta[:X.size(0)])
            loss = criterion(output, y)
            opt.zero_grad()

            if args.enable_apex:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

            if i % 10 == 0:
                logger.info("Iter: [{:d}][{:d}/{:d}]\t"
                             "Loss {:.3f} ({:.3f})\t"
                             "Prec@1 {:.3f} ({:.3f})\t".format(
                                epoch,
                                i,
                                len(train_loader),
                                loss.item(),
                                train_loss/train_n,
                                (output.max(1)[1] == y).sum().item() / y.size(0),
                                train_acc/train_n)
                )

        if args.early_stop:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.2:
                break
            prev_robust_acc = robust_acc
            best_state_dict = copy.deepcopy(model.state_dict())
        
    
        if args.random_precision_training and args.num_bits_schedule is not None:
            args.num_bits = args.num_bits_schedule[0]

            logger.info('Evaluating with PGD Attack under the lowest precision...')
            pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 20, 1, logger, args)
        
        else:
            logger.info('Evaluating with PGD Attack...')
            pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 20, 1, logger, args)


        logger.info('Evaluating with standard images...')
        test_loss, test_acc = evaluate_standard(test_loader, model, args)

        if pgd_acc > best_pgd_acc:
            best_pgd_acc = pgd_acc
            test_acc_best_pgd = test_acc

            best_state = {}
            best_state['state_dict'] = copy.deepcopy(model.state_dict())
            best_state['opt'] = copy.deepcopy(opt.state_dict())
            best_state['scheduler'] = copy.deepcopy(scheduler.state_dict())
            best_state['epoch'] = epoch 
            best_state['pgd_acc'] = pgd_acc
            best_state['best_pgd_acc'] = best_pgd_acc
            best_state['standard_acc'] = test_acc

            torch.save(best_state, os.path.join(args.save_dir, 'model.pth'))

        logger.info('Test Loss: %.4f  \t Test Acc: %.4f  \n PGD Loss: %.4f \t PGD Acc: %.4f \n Best PGD Acc: %.4f \t Test Acc with best PGD acc: %.4f', 
                    test_loss, test_acc, pgd_loss, pgd_acc, best_pgd_acc, test_acc_best_pgd)

        if epoch % 10 == 0:
            state = {}
            state['state_dict'] = model.state_dict()
            state['opt'] = opt.state_dict()
            state['scheduler'] = scheduler.state_dict()
            state['epoch'] = epoch 
            state['pgd_acc'] = pgd_acc
            state['best_pgd_acc'] = best_pgd_acc
            state['standard_acc'] = test_acc

            torch.save(state, os.path.join(args.save_dir, 'model_%d.pth' % epoch))

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        # logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
        #     epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)

    train_time = time.time()

    torch.save(best_state, os.path.join(args.save_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    if args.switchable_bn:
        num_bits_list = list(range(args.num_bits_schedule[0], args.num_bits_schedule[1]+1))
        model_test = torch.nn.DataParallel(net(num_bits_list, num_classes=args.num_classes, normalize=dataset_normalization).cuda())
    else:
        model_test = torch.nn.DataParallel(net(num_classes=args.num_classes, normalize=dataset_normalization).cuda())

    model_test.load_state_dict(best_state['state_dict'])
    model_test.float()
    model_test.eval()


    logger.info('Evaluating with PGD Attack...')

    if args.random_precision_training and args.num_bits_schedule is not None:
        args.num_bits = args.num_bits_schedule[0]

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 20, 1, logger, args)

    logger.info('Evaluating with standard images...')
    test_loss, test_acc = evaluate_standard(test_loader, model_test, args)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


    if args.num_bits_schedule is not None and args.switchable_bn:
        num_bits_list = list(range(args.num_bits_schedule[0], args.num_bits_schedule[1]+1))
    else:
        num_bits_list = [3,4,5,6,7,8,9,10]

    logger.info('Evaluating RPS with standard images...')
    test_loss, test_acc = evaluate_standard_random(test_loader, model_test, args, num_bits_list)

    logger.info('Evaluating RPS with PGD Attack...')
    # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)

    pgd_loss, pgd_acc = evaluate_pgd_random(test_loader, model_test, 20, 1, logger, args, num_bits_list=num_bits_list, num_round=10)

    logger.info('RPS Standard Acc: %.4f \t RPS PGD Acc: %.4f', test_acc, pgd_acc)



if __name__ == "__main__":
    main()
