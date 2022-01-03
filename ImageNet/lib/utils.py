import logging
import os
import datetime
import torchvision.models as models
import math
import torch
import yaml
from easydict import EasyDict
import shutil
import numpy as np 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30./n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def initiate_logger(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S',)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(output_path, 'log.txt'),'a+'))
    logger.info(pad_str(' LOGISTICS '))
    logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logger.info('Output Name: {}'.format(output_path))
    logger.info('User: {}'.format(os.getenv('USER')))
    return logger

def get_model_names():
	return sorted(name for name in models.__dict__
    		if name.islower() and not name.startswith("__")
    		and callable(models.__dict__[name]))

def pad_str(msg, total_len=70):
    rem_len = total_len - len(msg)
    return '*'*int(rem_len/2) + msg + '*'*int(rem_len/2)\

def parse_config_file(args):
    with open(args.config) as f:
        config = EasyDict(yaml.load(f))
        
    # Add args parameters to the dict
    for k, v in vars(args).items():
        config[k] = v
        
    # Add the output path
    config.output_name = 'ckpt/{:s}_def-{:s}_step{:d}_eps{:d}_repeat{:d}'.format(args.output_prefix, config.ADV.defense_method,
                         int(config.ADV.fgsm_step), int(config.ADV.clip_eps), config.ADV.n_repeats)

    return config


def save_checkpoint(state, is_best, filepath, epoch):
    filename = os.path.join(filepath, 'model_%d.pth' % epoch)
    # Save model
    torch.save(state, filename)

    # Save latest model
    shutil.copyfile(filename, os.path.join(filepath, 'model_latest.pth'))

    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model.pth'))


def set_random_precision(args, _iters, logger):
    num_bits_range = np.arange(args.num_bits_schedule[0], args.num_bits_schedule[1]+1)
    args.num_bits = np.random.choice(num_bits_range)

    if _iters % 10 == 0:
        logger.info('Random Precision: Iter [{}] num_bits = {}'.format(_iters, args.num_bits))