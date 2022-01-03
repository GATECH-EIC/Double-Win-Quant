import torch as ch
from torchvision import models, transforms, datasets
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
import argparse
import json
import pdb

#####
import argparse
import copy
import logging
import os
import time


import sys
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from apex import amp

from model.preact_resnet import PreActResNet18
from model.wide_resnet import WideResNet32
from model.mobilenetv2 import MobileNetV2

from utils import (upper_limit, lower_limit, std, clamp,
    attack_pgd, evaluate_pgd, evaluate_standard)

from utils import set_random_precision, evaluate_pgd_random, evaluate_standard_random, evaluate_pgd_heter_prec, evaluate_pgd_heter_prec_random

from tqdm import tqdm
from torchvision.utils import save_image

logger = logging.getLogger(__name__)

torch.multiprocessing.set_start_method('spawn', force=True)

bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
#####



ch.set_default_tensor_type('torch.cuda.FloatTensor')

def norm(t):
    assert len(t.shape) == 4
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*ch.exp(lr*g)
    neg = (1-real_x)*ch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def linf_step(x, g, lr):
    return x + lr*ch.sign(g)

def l2_prior_step(x, g, lr):
    new_x = x + lr*g/norm(g)
    norm_new_x = norm(new_x)
    norm_mask = (norm_new_x < 1.0).float()
    return new_x*norm_mask + (1-norm_mask)*new_x/norm_new_x

def gd_prior_step(x, g, lr):
    return x + lr*g
   
def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + ch.clamp(new_x - orig, -eps, eps)
    return proj

##
# Main functions
##

def make_adversarial_examples(attackbit, evalbit,image, true_label, args, model_to_fool, IMAGENET_SL):
    '''
    The main process for generating adversarial examples with priors.
    '''
    # Initial setup

    model_to_fool.module.set_precision(num_bits=np.random.choice(args.num_attack_bits_list), num_grad_bits=0)


    prior_size = IMAGENET_SL if not args.tiling else args.tile_size
    upsampler = Upsample(size=(IMAGENET_SL, IMAGENET_SL))
    total_queries = ch.zeros(args.batch_size)
    prior = ch.zeros(args.batch_size, 3, prior_size, prior_size)
    dim = prior.nelement()/args.batch_size
    prior_step = gd_prior_step if args.mode == 'l2' else eg_step
    image_step = l2_image_step if args.mode == 'l2' else linf_step
    proj_maker = l2_proj if args.mode == 'l2' else linf_proj
    proj_step = proj_maker(image, args.epsilon)
    # print(image.max(), image.min())

    # Loss function
    criterion = ch.nn.CrossEntropyLoss(reduction='none')
    def normalized_eval(x):
        x_copy = x.clone()
        x_copy = ch.stack([F.normalize(x_copy[i], [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]) \
                        for i in range(args.batch_size)])
        return model_to_fool(x_copy)

    L = lambda x: criterion(normalized_eval(x), true_label)
    losses = L(image)

    # Original classifications
    orig_images = image.clone()
    # !!!!!!!!!!!
    orig_classes = normalized_eval(image).argmax(1).cuda()
    correct_classified_mask = (orig_classes == true_label).float()
    print("original acc: %f", correct_classified_mask.sum().item()/args.batch_size)
    total_ims = correct_classified_mask.sum()
    not_dones_mask = correct_classified_mask.clone()

    t = 0
    while not ch.any(total_queries > args.max_queries):
        t += args.gradient_iters*2
        if t >= args.max_queries:
            break
        if not args.nes:
            ## Updating the prior: 
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = args.exploration*ch.randn_like(prior)/(dim**0.5) 
            # Query deltas for finite difference estimator
            q1 = upsampler(prior + exp_noise)
            q2 = upsampler(prior - exp_noise)
            # Loss points for finite difference estimator
            l1 = L(image + args.fd_eta*q1/norm(q1)) # L(prior + c*noise)
            l2 = L(image + args.fd_eta*q2/norm(q2)) # L(prior - c*noise)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2)/(args.fd_eta*args.exploration)
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1)*exp_noise
            # Update the prior with the estimated gradient
            prior = prior_step(prior, est_grad, args.online_lr)
        else:
            prior = ch.zeros_like(image)
            for _ in range(args.gradient_iters):
                exp_noise = ch.randn_like(image)/(dim**0.5) 
                est_deriv = (L(image + args.fd_eta*exp_noise) - L(image - args.fd_eta*exp_noise))/args.fd_eta
                prior += est_deriv.view(-1, 1, 1, 1)*exp_noise

            # Preserve images that are already done, 
            # Unless we are specifically measuring gradient estimation
            prior = prior*not_dones_mask.view(-1, 1, 1, 1)

        ## Update the image:
        # take a pgd step using the prior
        new_im = image_step(image, upsampler(prior*correct_classified_mask.view(-1, 1, 1, 1)), args.image_lr)
        image = proj_step(new_im)
        image = ch.clamp(image, 0, 1)
        if args.mode == 'l2':
            if not ch.all(norm(image - orig_images) <= args.epsilon + 1e-3):
                pdb.set_trace()
        else:
            if not (image - orig_images).max() <= args.epsilon + 1e-3:
                pdb.set_trace()

        ## Continue query count
        total_queries += 2*args.gradient_iters*not_dones_mask
        model_to_fool.module.set_precision(num_bits=np.random.choice(args.num_bits_list), num_grad_bits=0)
        not_dones_mask = not_dones_mask*((normalized_eval(image).argmax(1) == true_label).float())
        model_to_fool.module.set_precision(num_bits=np.random.choice(args.num_attack_bits_list), num_grad_bits=0)

        ## Logging stuff
        new_losses = L(image)
        success_mask = (1 - not_dones_mask)*correct_classified_mask
        num_success = success_mask.sum()
                
        # if (num_success > 10):
        #     print(success_mask.nonzero())
        #     for i in range(len(success_mask.nonzero())):
        #         save_image(orig_images[success_mask.nonzero()[i].item()], "./images/{}ori.png".format(i))
        #         save_image(image[success_mask.nonzero()[i].item()], "./images/{}success.png".format(i))
        #     exit()
        current_success_rate = (num_success/correct_classified_mask.sum()).cpu().item()
        success_queries = ((success_mask*total_queries).sum()/num_success).cpu().item()
        not_done_loss = ((new_losses*not_dones_mask).sum()/not_dones_mask.sum()).cpu().item()
        max_curr_queries = total_queries.max().cpu().item()
        if args.log_progress:
            print("Queries: %d | Success rate: %f | acc: %f | Average queries: %f" % (max_curr_queries, current_success_rate, 
            (1 - current_success_rate)*correct_classified_mask.sum().item()/args.batch_size, success_queries))

        if current_success_rate == 1.0:
            break

    return {
            'average_queries': success_queries,
            'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
            'success_rate': current_success_rate,
            'images_orig': orig_images.cpu().numpy(),
            'images_adv': image.cpu().numpy(),
            'all_queries': total_queries.cpu().numpy(),
            'correctly_classified': correct_classified_mask.cpu().numpy(),
            'success': success_mask.cpu().numpy(),
            'correct': (1 - current_success_rate)*correct_classified_mask.sum().item()
    }

def get_loaders(dir_, batch_size, dataset='cifar10'):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)
    else:
        print('Wrong dataset:', dataset)
        exit()

    train_loader = ch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    test_loader = ch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=2,
    )
    return train_loader, test_loader

def main(test_loader, args, model_to_fool, dataset_size):
    total_correct, total_adv, total_queries, correct, total = 0, 0, 0, 0, 0
    with ch.no_grad():
        pbar = tqdm(enumerate(test_loader), file=sys.stdout, bar_format=bar_format, ncols=80)
        for i, (images, targets) in pbar:
            print(i)
            if i*args.batch_size >= 10000 - args.batch_size:
                break
            res = make_adversarial_examples(np.random.choice(args.num_attack_bits_list),
            np.random.choice(args.num_bits_list),
            images.cuda(), targets.cuda(), args, model_to_fool, dataset_size)
            ncc = res['num_correctly_classified'] # Number of correctly classified images (originally)
            num_adv = ncc * res['success_rate'] # Success rate was calculated as (# adv)/(# correct classified)
            queries = num_adv * res['average_queries'] # Average queries was calculated as (total queries for advs)/(# advs)
            total_correct += ncc
            total_adv += num_adv
            total_queries += queries
            correct += res['correct']
            total += args.batch_size
            print("correct:", correct)
            print("sofar acc:", correct / total)


    print("-"*80)
    print("Final Success Rate: {succ} | Final Average Queries: {aq}".format(
            aq=total_queries/total_adv,
            succ=total_adv/total_correct))
    print("correct:", correct)
    print("final acc:", correct / total)
    print("-"*80)

class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params
    
    def __getattr__(self, x):
        return self.params[x.lower()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ######
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='/home/yf22/dataset', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--network', default='PreActResNet18', choices=['PreActResNet18', 'WideResNet32', 'MobileNetV2'])
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
    ######

    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--fd-eta', type=float, help='\eta, used to estimate the derivative via finite differences')
    parser.add_argument('--image-lr', type=float, help='Learning rate for the image (iterative attack)')
    parser.add_argument('--online-lr', type=float, help='Learning rate for the prior')
    parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior')
    parser.add_argument('--tile-size', type=int, help='the side length of each tile (for the tiling prior)')
    parser.add_argument('--json-config', type=str, help='a config file to be passed in instead of arguments')
    parser.add_argument('--epsilon', type=float, help='the lp perturbation bound')
    # parser.add_argument('--batch-size', type=int, help='batch size for bandits')
    parser.add_argument('--log-progress', action='store_true')
    parser.add_argument('--nes', action='store_true')
    parser.add_argument('--tiling', action='store_true')
    parser.add_argument('--gradient-iters', type=int)
    parser.add_argument('--total-images', type=int)
    args = parser.parse_args()

    args_dict = None
    if not args.json_config:
        # If there is no json file, all of the args must be given
        args_dict = vars(args)
    else:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None or k == "num_attack_bits_list"}
        defaults.update(arg_vars)
        args = Parameters(defaults)
        args_dict = defaults
    
    ######
    assert type(args.pretrain) == str and os.path.exists(args.pretrain)
    assert args.num_bits_list is not None
    # print(args_dict)
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

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset)

    if args.network == 'PreActResNet18':
        net = PreActResNet18
    elif args.network == 'WideResNet32':
        net = WideResNet32
    elif args.network == 'MobileNetV2':
        net = MobileNetV2
    else:
        print('Wrong network:', args.network)

    if args.switchable_bn:
        model = net(args.num_bits_list, num_classes=args.num_classes).cuda()
    else:
        model = net(num_classes=args.num_classes).cuda()

    model = torch.nn.DataParallel(model)

    pretrained_model = torch.load(args.pretrain)
    partial = pretrained_model['state_dict']

    model.load_state_dict(partial)

    start_epoch = pretrained_model['epoch'] + 1

    print('Resume from Epoch %d. Load pretrained weight.' % start_epoch)
    ######

    model.eval()
    
    main(test_loader, args, model, 32)
