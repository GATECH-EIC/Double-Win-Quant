# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import init_paths
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from utils import *
from validation import validate, validate_pgd, validate_pgd_random, validate_random
import torchvision.models as models
from resnet import ResNet50


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default='/data1/ILSVRC/Data/CLS-LOC/', help='path to dataset')
    parser.add_argument('--output_prefix', default='free_adv', type=str,
                    help='prefix used to define output path')
    parser.add_argument('-c', '--config', default='config.yml', type=str, metavar='Path',
                    help='path to the config file (default: config.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    
    parser.add_argument('--automatic_resume', action='store_true',
                    help='automatically resume from the latest checkpoint')

    parser.add_argument('--num_bits',default=None,type=int,
                        help='num bits for weight and activation')

    parser.add_argument('--random_precision_training', action='store_true',
                        help='enble random precision training')

    parser.add_argument('--switchable_bn', action='store_true',
                        help='if use switchable BN')               
    parser.add_argument('--num_bits_schedule', default=None, type=int, nargs='*',
                        help='precision schedule for weight/act in random precision training')

    parser.add_argument('--lr',default=None,type=float,
                        help='initial learning rate')
    parser.add_argument('--epochs',default=None,type=int,
                        help='total training epochs')
    parser.add_argument('--batch_size',default=None,type=int,
                        help='training batch size')
    parser.add_argument('--clip_eps',default=None,type=float,
                        help='epsilon')
    parser.add_argument('--fgsm_step',default=None,type=float,
                        help='step size (alpha)')
    parser.add_argument('--defense_method',default=None,type=str,
                        help='pgd/random/free')
    return parser.parse_args()


# Parase config file and initiate logging
config = parse_config_file(parse_args())
logger = initiate_logger(config.output_name)
print = logger.info
cudnn.benchmark = True


if config.lr is not None:
    config.TRAIN.lr = config.lr
if config.epochs is not None:
    config.TRAIN.epochs = config.epochs
if config.batch_size is not None:
    config.DATA.batch_size = config.batch_size
if config.clip_eps is not None:
    config.ADV.clip_eps = config.clip_eps
if config.fgsm_step is not None:
    config.ADV.fgsm_step = config.fgsm_step
if config.defense_method is not None:
    config.ADV.defense_method = config.defense_method


if config.random_precision_training:
    if config.num_bits_schedule is None:
        print('Please specify num_bits_schedule.')
        exit()
    else:
        config.num_bits_list = list(range(config.num_bits_schedule[0], config.num_bits_schedule[1]+1))

else:
    config.num_bits_list = [3,4,5,6,7,8,9,10]


def main():
    # Scale and initialize the parameters
    best_pgd_acc = 0
    best_epoch = 0
    standard_acc_at_best_pgd = 0

    if config.ADV.defense_method != 'free':
        config.ADV.n_repeats = 1

    config.TRAIN.epochs = int(math.ceil(config.TRAIN.epochs / config.ADV.n_repeats))
    config.ADV.fgsm_step /= config.DATA.max_color_value
    config.ADV.clip_eps /= config.DATA.max_color_value
    
    # Create output folder
    if not os.path.isdir(config.output_name):
        os.makedirs(config.output_name)
    
    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in config.items(): print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    if config.automatic_resume:
        model_path = os.path.join(config.output_name, 'model_latest.pth')
        if os.path.isfile(model_path):
            config.pretrained = False

    if config.switchable_bn:
        assert config.num_bits_schedule is not None

        model = ResNet50(num_bits_list=config.num_bits_list, pretrained=config.pretrained).cuda()
    else:
        model = ResNet50(pretrained=config.pretrained).cuda()


    # Wrap the model into DataParallel
    model = torch.nn.DataParallel(model)
    
    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), config.TRAIN.lr,
                                momentum=config.TRAIN.momentum,
                                weight_decay=config.TRAIN.weight_decay)
    
    if config.automatic_resume:
        model_path = os.path.join(config.output_name, 'model_latest.pth')

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            config.TRAIN.start_epoch = checkpoint['epoch'] + 1
            best_pgd_acc = checkpoint['best_pgd_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Automatic resume from '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
        else:
            print("=> Automatic resume failed, no such latest model: '{}'".format(model_path)) 

    # Resume if a valid checkpoint path is provided
    if config.resume:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            config.TRAIN.start_epoch = checkpoint['epoch'] + 1
            best_pgd_acc = checkpoint['best_pgd_acc']
            standard_acc_at_best_pgd = checkpoint['standard_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

            
    # Initiate data loaders
    traindir = os.path.join(config.data, 'train')
    valdir = os.path.join(config.data, 'val')
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(config.DATA.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.DATA.batch_size, shuffle=True,
        num_workers=config.DATA.workers, pin_memory=True, sampler=None)
    
    normalize = transforms.Normalize(mean=config.TRAIN.mean,
                                    std=config.TRAIN.std)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(config.DATA.img_size),
            transforms.CenterCrop(config.DATA.crop_size),
            transforms.ToTensor(),
        ])),
        batch_size=config.DATA.batch_size, shuffle=False,
        num_workers=config.DATA.workers, pin_memory=True)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if config.evaluate:
        logger.info(pad_str(' Performing PGD Attacks '))
        validate_random(val_loader, model, criterion, config, logger)

        for pgd_param in config.ADV.pgd_attack:
            validate_pgd_random(val_loader, model, criterion, pgd_param[0], pgd_param[1], config, logger)
        return
    
    
    for epoch in range(config.TRAIN.start_epoch, config.TRAIN.epochs):
        adjust_learning_rate(config.TRAIN.lr, optimizer, epoch, config.ADV.n_repeats)

        # train for one epoch
        if config.ADV.defense_method == 'free':
            train_free(train_loader, model, criterion, optimizer, epoch)
        elif config.ADV.defense_method == 'pgd':
            train_pgd(train_loader, model, criterion, optimizer, epoch)
        elif config.ADV.defense_method == 'random':
            train_fgsm_random(train_loader, model, criterion, optimizer, epoch)
        else:
            print('Wrong defense method:%s', config.ADV.defense_method)
            exit()

        # evaluate on validation set
        if config.random_precision_training and config.num_bits_schedule is not None:
            config.num_bits = config.num_bits_schedule[0]

        if epoch % 3 == 0:
            pgd_acc = validate_pgd_random(val_loader, model, criterion, 10, 1.0/255.0, config, logger)
            standard_acc = validate_random(val_loader, model, criterion, config, logger)

            # remember best prec@1 and save checkpoint
            is_best = pgd_acc > best_pgd_acc
            best_pgd_acc = max(pgd_acc, best_pgd_acc)

            if is_best:
                best_epoch = epoch
                standard_acc_at_best_pgd = standard_acc

            save_checkpoint({
                'epoch': epoch,
                'arch': config.TRAIN.arch,
                'state_dict': model.state_dict(),
                'best_pgd_acc': best_pgd_acc,
                'standard_acc': standard_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, config.output_name, epoch=epoch)

            print('Current Best PGD Acc:%f, Achieved at %d Epoch, with Standard Acc: %f', best_pgd_acc, best_epoch, standard_acc_at_best_pgd)
        
    # Automatically perform PGD Attacks at the end of training
    logger.info(pad_str(' Performing PGD Attacks '))

    if config.random_precision and config.num_bits_schedule is not None:
        config.num_bits = config.num_bits_schedule[0]

    for pgd_param in config.ADV.pgd_attack:
        validate_pgd_random(val_loader, model, criterion, pgd_param[0], pgd_param[1], config, logger)

        
# Free Adversarial Training Module        
global global_noise_data
global_noise_data = torch.zeros([config.DATA.batch_size, 3, config.DATA.crop_size, config.DATA.crop_size]).cuda()
def train_free(train_loader, model, criterion, optimizer, epoch):
    global global_noise_data
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)
        for j in range(config.ADV.n_repeats):
            _iters = epoch * len(train_loader) + i

            if config.random_precision_training:
                set_random_precision(config, _iters, logger)

            model.module.set_precision(config.num_bits, 0)

            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)
            loss = criterion(output, target)
            
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, config.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-config.ADV.clip_eps, config.ADV.clip_eps)

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TRAIN.print_freq == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
                sys.stdout.flush()


def train_pgd(train_loader, model, criterion, optimizer, epoch):
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        _iters = epoch * len(train_loader) + i

        if config.random_precision_training:
            set_random_precision(config, _iters, logger)

        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        model.module.set_precision(config.num_bits, 0)

        delta = torch.zeros_like(input).cuda()
        if config.ADV.delta_init == 'random':
            delta.uniform_(-config.ADV.clip_eps, config.ADV.clip_eps)
        delta.requires_grad = True

        for _ in range(config.ADV.attack_iters):
            in1 = input + delta
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)

            loss = criterion(output, target)
            loss.backward()

            grad = delta.grad.detach()
            delta.data = delta.data + config.ADV.fgsm_step * torch.sign(grad)
            delta.data.clamp_(-config.ADV.clip_eps, config.ADV.clip_eps)
            delta.grad.zero_()

        delta = delta.detach()

        model.module.set_precision(config.num_bits, 0)

        in1 = input + delta
        in1.clamp_(0, 1.0)
        in1.sub_(mean).div_(std)
        output = model(in1)
        loss = criterion(output, target)
        
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.TRAIN.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
            sys.stdout.flush()


def train_fgsm_random(train_loader, model, criterion, optimizer, epoch):
    mean = torch.Tensor(np.array(config.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,config.DATA.crop_size, config.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(config.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.DATA.crop_size, config.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        _iters = epoch * len(train_loader) + i

        if config.random_precision_training:
            set_random_precision(config, _iters, logger)

        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        model.module.set_precision(config.num_bits, 0)

        delta = torch.zeros_like(input).cuda()
        delta.uniform_(-config.ADV.clip_eps, config.ADV.clip_eps)
        delta.requires_grad = True

        in1 = input + delta
        in1.clamp_(0, 1.0)
        in1.sub_(mean).div_(std)
        output = model(in1)

        loss = criterion(output, target)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = delta.data + config.ADV.fgsm_step * torch.sign(grad)
        delta.data.clamp_(-config.ADV.clip_eps, config.ADV.clip_eps)

        delta = delta.detach()

        model.module.set_precision(config.num_bits, 0)

        in1 = input + delta
        in1.clamp_(0, 1.0)
        in1.sub_(mean).div_(std)
        output = model(in1)
        loss = criterion(output, target)
        
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.TRAIN.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
            sys.stdout.flush()


if __name__ == '__main__':
    main()